"""Public entry point for HPO-Agent."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from dataclasses import replace as dataclass_replace
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool

from hpo_agent.models import HPOConfig, HPOResult, ParamSpace, ParamSpec
from hpo_agent.prompts import (
    EXPERT_AGENT_DEFAULT_PROMPT,
    PARAM_SPACE_COMPLETION_PROMPT,
    PARAM_SPACE_GENERATION_PROMPT,
    SUPERVISOR_DEFAULT_PROMPT,
    build_system_prompt,
)
from hpo_agent.providers import (
    AnthropicLLMProvider,
    GoogleLLMProvider,
    LLMProviderBase,
    OpenAILLMProvider,
)
from hpo_agent.report import ReportGenerator
from hpo_agent.supervisor import Supervisor
from hpo_agent.tools import (
    BayesianOptimizationTool,
    ChangeSearchSpaceTool,
    ExpertAgentTool,
    SobolSearchTool,
)

logger = logging.getLogger(__name__)


class HPOAgent:
    """ユーザーに公開するエントリーポイント。

    依存解決（LLM プロバイダー・ツール）を行い、
    Supervisor を構築して実行する。

    eval_fn のシグネチャは (params: dict) -> float。
    モデルの学習・評価はすべて eval_fn 内で実装する。

    Args:
        eval_fn: 評価関数。パラメータ辞書を受け取りスコア（大きいほど良い）を返す。
            シグネチャ: (params: dict) -> float。
        n_trials: 総試行回数。
        param_space: 最適化対象パラメータ空間。省略時は LLM が自動生成する。
        seed: 乱数シード。None の場合は非決定的。
        prompts: エージェント別追加プロンプト辞書。キー: "supervisor", "expert_agent"。
        llm_model: LLM モデル名（.env 設定の上書き用）。
    """

    def __init__(
        self,
        eval_fn: Callable[[dict[str, Any]], float],
        n_trials: int,
        param_space: ParamSpace | None = None,
        seed: int | None = None,
        prompts: dict[str, str] | None = None,
        llm_model: str | None = None,
    ) -> None:
        """HPOAgent を初期化する。"""
        self._config = HPOConfig(
            eval_fn=eval_fn,
            n_trials=n_trials,
            param_space=param_space,
            seed=seed,
            prompts=prompts or {},
            llm_model=llm_model,
        )

    def run(self) -> HPOResult:
        """HPO を実行して結果を返す。

        param_space が未指定の場合は LLM が自動生成する。

        Returns:
            最適化結果（HPOResult）。
        """
        param_space = self._config.param_space
        generated_param_space: ParamSpace | None = None
        if param_space is None:
            logger.info("Supervisor agent creates search space.\n")
            param_space = self._generate_param_space()
            generated_param_space = param_space
        elif any(s.is_partial for s in param_space.specs):
            logger.info("Supervisor agent completes partial search space.\n")
            param_space = self._complete_param_space(param_space)
            generated_param_space = param_space
        supervisor = self._build_supervisor(param_space, generated_param_space)
        return supervisor.run(self._config)

    def _generate_param_space(self) -> ParamSpace:
        """LLM を使ってパラメータ空間を自動生成する。

        eval_fn のソースコード・試行回数を LLM に渡し、
        ParamSpaceSchema として構造化出力を受け取る。

        Returns:
            LLM が生成した ParamSpace。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from hpo_agent.models import ParamSpaceSchema

        llm_provider = self._resolve_llm_provider()
        llm = llm_provider.get_llm(temperature=0)
        structured_llm = llm.with_structured_output(ParamSpaceSchema)

        try:
            eval_fn_source = inspect.getsource(self._config.eval_fn)
        except (OSError, TypeError):
            eval_fn_source = "(ソース取得不可)"

        prompt = PARAM_SPACE_GENERATION_PROMPT.format(
            n_trials=self._config.n_trials,
            eval_fn_source=eval_fn_source,
        )
        result: ParamSpaceSchema = structured_llm.invoke(  # type: ignore[assignment]
            [
                SystemMessage(content="あなたは機械学習の専門家です。"),
                HumanMessage(content=prompt),
            ]
        )
        param_space = result.to_param_space()
        logger.info(
            "[HPOAgent] Auto-generated parameter space by LLM (n_trials=%d):\n%s",
            self._config.n_trials,
            "\n".join(self._format_param_space(param_space)),
        )
        return param_space

    def _complete_param_space(self, partial_space: ParamSpace) -> ParamSpace:
        """部分指定の ParamSpec を LLM で補完して完全な ParamSpace を返す。

        partial_space に含まれる部分指定 spec（is_partial=True）の範囲・選択肢を
        LLM に補完させ、完全指定 spec と結合して返す。

        Args:
            partial_space: 部分指定の ParamSpec を含む ParamSpace。

        Returns:
            すべての ParamSpec が完全指定された ParamSpace。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from hpo_agent.models import ParamSpaceSchema

        partial_specs = [s for s in partial_space.specs if s.is_partial]
        complete_specs = [s for s in partial_space.specs if not s.is_partial]

        try:
            eval_fn_source = inspect.getsource(self._config.eval_fn)
        except (OSError, TypeError):
            eval_fn_source = "(ソース取得不可)"

        partial_specs_description = "\n".join(
            self._format_partial_spec_line(s) for s in partial_specs
        )
        complete_specs_description = (
            "\n".join(self._format_param_space(ParamSpace(specs=tuple(complete_specs))))
            if complete_specs
            else "(なし)"
        )

        prompt = PARAM_SPACE_COMPLETION_PROMPT.format(
            n_trials=self._config.n_trials,
            eval_fn_source=eval_fn_source,
            partial_specs_description=partial_specs_description,
            complete_specs_description=complete_specs_description,
        )

        llm_provider = self._resolve_llm_provider()
        llm = llm_provider.get_llm(temperature=0)
        structured_llm = llm.with_structured_output(ParamSpaceSchema)
        result: ParamSpaceSchema = structured_llm.invoke(  # type: ignore[assignment]
            [
                SystemMessage(content="あなたは機械学習の専門家です。"),
                HumanMessage(content=prompt),
            ]
        )
        original_partial_map: dict[str, ParamSpec] = {s.name: s for s in partial_specs}
        completed_specs: list[ParamSpec] = [
            (
                self._enforce_user_bounds(
                    s.to_param_spec(), original_partial_map[s.name]
                )
                if s.name in original_partial_map
                else s.to_param_spec()
            )
            for s in result.specs
        ]
        all_specs = tuple(complete_specs) + tuple(completed_specs)
        param_space = ParamSpace(specs=all_specs)
        logger.info(
            "[HPOAgent] Completed partial param space by LLM (n_trials=%d):\n%s",
            self._config.n_trials,
            "\n".join(self._format_param_space(param_space)),
        )
        return param_space

    def _format_param_space(self, param_space: ParamSpace) -> list[str]:
        """ParamSpace の各 spec を人が読める形式の行リストとして返す。"""
        lines: list[str] = []
        for spec in param_space.specs:
            if spec.type == "categorical":
                lines.append(
                    f"  - {spec.name}: categorical, choices={list(spec.choices or [])}"
                )
            else:
                scale = "log" if spec.log else "linear"
                lines.append(
                    f"  - {spec.name}: {spec.type},"
                    f" [{spec.low}, {spec.high}], {scale}"
                )
        return lines

    @staticmethod
    def _format_partial_spec_line(spec: ParamSpec) -> str:
        """部分指定 ParamSpec を LLM への説明行に変換する。

        ユーザーが指定済みの low / high は「ユーザー指定・変更不可」と明示する。
        """
        parts = [f"- {spec.name}: {spec.type}"]
        if spec.low is not None:
            parts.append(f"low={spec.low} (ユーザー指定・変更不可)")
        if spec.high is not None:
            parts.append(f"high={spec.high} (ユーザー指定・変更不可)")
        return ", ".join(parts)

    @staticmethod
    def _enforce_user_bounds(llm_spec: ParamSpec, original: ParamSpec) -> ParamSpec:
        """LLM が返した spec にユーザー指定の low / high を上書きして返す。

        Args:
            llm_spec: LLM が補完した ParamSpec。
            original: ユーザーが指定した部分指定 ParamSpec。

        Returns:
            ユーザー指定の bound を優先した ParamSpec。
        """
        low = original.low if original.low is not None else llm_spec.low
        high = original.high if original.high is not None else llm_spec.high
        return dataclass_replace(llm_spec, low=low, high=high)

    def _resolve_llm_provider(self) -> LLMProviderBase:
        """環境変数から LLM プロバイダーを構築して返す。

        LLM_PROVIDER 環境変数でプロバイダーを切り替える。
        未設定または "google" の場合は GoogleLLMProvider を返す。
        "openai" の場合は OpenAILLMProvider を返す。
        "anthropic" の場合は AnthropicLLMProvider を返す。

        Raises:
            ValueError: 未サポートの LLM_PROVIDER 値が指定された場合。
        """
        load_dotenv()
        api_key = os.environ["LLM_API_KEY"]
        model_name = self._config.llm_model or os.environ["LLM_MODEL_NAME"]
        provider = os.getenv("LLM_PROVIDER", "google")
        if provider == "google":
            return GoogleLLMProvider(api_key=api_key, model_name=model_name)
        elif provider == "openai":
            return OpenAILLMProvider(api_key=api_key, model_name=model_name)
        elif provider == "anthropic":
            return AnthropicLLMProvider(api_key=api_key, model_name=model_name)
        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: '{provider}'. "
                "Supported values: 'google', 'openai', 'anthropic'."
            )

    def _build_supervisor(
        self,
        param_space: ParamSpace,
        generated_param_space: ParamSpace | None = None,
    ) -> Supervisor:
        """Supervisor インスタンスを構築して返す。

        Args:
            param_space: 使用するパラメータ空間。
            generated_param_space: LLM が自動生成したパラメータ空間。レポートに記載される。

        Returns:
            構築済みの Supervisor。
        """
        llm_provider = self._resolve_llm_provider()
        supervisor_llm = llm_provider.get_llm(temperature=0)
        expert_llm = llm_provider.get_llm(temperature=0.3)
        seed = self._config.seed
        eval_fn = self._config.eval_fn

        expert_system_prompt = build_system_prompt(
            EXPERT_AGENT_DEFAULT_PROMPT,
            self._config.prompts.get("expert_agent"),
        )
        tools: list[BaseTool] = [
            SobolSearchTool(
                eval_fn=eval_fn,
                param_space=param_space,
                seed=seed,
                name="sobol_search",
                description="Sobol 列による準ランダム探索",
            ),
            BayesianOptimizationTool(
                eval_fn=eval_fn,
                param_space=param_space,
                seed=seed,
                name="bayesian_optimization",
                description="Optuna を用いたベイズ最適化による探索",
            ),
            ExpertAgentTool(
                eval_fn=eval_fn,
                param_space=param_space,
                llm=expert_llm,
                system_prompt=expert_system_prompt,
                name="expert_agent",
                description="専門家 AI エージェントによる決め打ち探索",
            ),
            ChangeSearchSpaceTool(
                param_space=param_space,
                name="change_search_space",
                description=(
                    "探索空間を変更する（狭め・拡大どちらも可能）。"
                    "有望な範囲が特定できた場合に絞り込み、"
                    "狭めすぎた場合や新たな領域を探索したい場合に拡大する。"
                    "param_updates に JSON 文字列で新しい範囲を指定する。"
                    '例: [{"name": "learning_rate", "low": 0.001, "high": 0.5}]'
                ),
            ),
        ]

        supervisor_prompt = build_system_prompt(
            SUPERVISOR_DEFAULT_PROMPT,
            self._config.prompts.get("supervisor"),
        )
        return Supervisor(
            llm=supervisor_llm,
            tools=tools,
            report_generator=ReportGenerator(),
            system_prompt=supervisor_prompt,
            generated_param_space=generated_param_space,
        )
