"""Public entry point for HPO-Agent."""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import Callable
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool

from hpo_agent.adapters import (
    LightGBMAdapter,
    ModelAdapterBase,
    PyTorchAdapter,
    SklearnAdapter,
)
from hpo_agent.models import HPOConfig, HPOResult, ParamSpace
from hpo_agent.prompts import (
    EXPERT_AGENT_DEFAULT_PROMPT,
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
    ExpertAgentTool,
    NarrowSearchSpaceTool,
    SobolSearchTool,
)

logger = logging.getLogger(__name__)


class HPOAgent:
    """ユーザーに公開するエントリーポイント。

    依存解決（アダプター・LLM プロバイダー・ツール）を行い、
    Supervisor を構築して実行する。

    LightGBM の場合:
        model に lgb.LGBMModel インスタンスを渡す。eval_fn のシグネチャは
        (model, X, y) -> float。X, y は必須。

    PyTorch の場合:
        model にモデルファクトリ関数 (params: dict) -> model を渡す。
        eval_fn のシグネチャは (model) -> float（学習・評価ループ全体を含む）。
        X, y は不要。

    Args:
        model: チューニング対象モデルまたはモデルファクトリ関数。
            LightGBM: lgb.LGBMModel のインスタンス。
            sklearn: sklearn.base.BaseEstimator のサブクラスインスタンス。
            PyTorch: パラメータ辞書を受け取りモデルを返す callable。
        eval_fn: 評価関数。大きいほど良いスコア。
            LightGBM / sklearn: (model, X, y) -> float。
            PyTorch: (model) -> float（学習・評価ループ全体）。
        n_trials: 総試行回数。
        X: 特徴量データ（LightGBM のみ使用）。
        y: ターゲットデータ（LightGBM のみ使用）。
        param_space: 最適化対象パラメータ空間。省略時は LLM が自動生成する。
        seed: 乱数シード。None の場合は非決定的。
        prompts: エージェント別追加プロンプト辞書。キー: "supervisor", "expert_agent"。
        llm_model: LLM モデル名（.env 設定の上書き用）。
    """

    def __init__(
        self,
        model: Any,
        eval_fn: Callable[..., float],
        n_trials: int,
        X: Any = None,
        y: Any = None,
        param_space: ParamSpace | None = None,
        seed: int | None = None,
        prompts: dict[str, str] | None = None,
        llm_model: str | None = None,
    ) -> None:
        """HPOAgent を初期化する。"""
        self._config = HPOConfig(
            model=model,
            eval_fn=eval_fn,
            n_trials=n_trials,
            X=X,
            y=y,
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
        adapter, param_space = self._resolve_adapter()
        generated_param_space: ParamSpace | None = None
        if param_space is None:
            param_space = self._generate_param_space()
            generated_param_space = param_space
        supervisor = self._build_supervisor(adapter, param_space, generated_param_space)
        return supervisor.run(self._config)

    def _resolve_adapter(self) -> tuple[ModelAdapterBase, ParamSpace | None]:
        """モデル型に応じたアダプターと使用するパラメータ空間を返す。

        param_space が未指定の場合は None を返す。呼び出し元で LLM 自動生成を行う。

        Returns:
            (adapter, param_space) のタプル。param_space は None の場合あり。

        Raises:
            TypeError: 未対応のモデル型の場合。
        """
        import lightgbm as lgb
        from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

        model = self._config.model
        adapter: ModelAdapterBase
        if isinstance(model, lgb.LGBMModel):
            adapter = LightGBMAdapter(
                model=model,
                eval_fn=self._config.eval_fn,
                X=self._config.X,
                y=self._config.y,
            )
        elif isinstance(model, BaseEstimator):
            adapter = SklearnAdapter(
                model=model,
                eval_fn=self._config.eval_fn,
                X=self._config.X,
                y=self._config.y,
            )
        elif callable(model):
            adapter = PyTorchAdapter(
                model_fn=model,
                eval_fn=self._config.eval_fn,
            )
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                "Supported: LightGBM (lgb.LGBMModel), scikit-learn (BaseEstimator subclasses), "
                "PyTorch (callable な model_fn)。"
            )

        return adapter, self._config.param_space

    def _generate_param_space(self) -> ParamSpace:
        """LLM を使ってモデルに適したパラメータ空間を自動生成する。

        eval_fn のソースコード・モデルクラス名・試行回数を LLM に渡し、
        ParamSpaceSchema として構造化出力を受け取る。

        Returns:
            LLM が生成した ParamSpace。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        from hpo_agent.models import ParamSpaceSchema

        llm_provider = self._resolve_llm_provider()
        llm = llm_provider.get_llm(temperature=0)
        structured_llm = llm.with_structured_output(ParamSpaceSchema)

        model_class_name = type(self._config.model).__name__
        try:
            eval_fn_source = inspect.getsource(self._config.eval_fn)
        except (OSError, TypeError):
            eval_fn_source = "(ソース取得不可)"

        prompt = PARAM_SPACE_GENERATION_PROMPT.format(
            model_class_name=model_class_name,
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
            "[HPOAgent] LLM が自動生成したパラメータ空間 (model=%s, n_trials=%d):\n%s",
            model_class_name,
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
        adapter: ModelAdapterBase,
        param_space: ParamSpace,
        generated_param_space: ParamSpace | None = None,
    ) -> Supervisor:
        """Supervisor インスタンスを構築して返す。

        Args:
            adapter: モデルアダプター。
            param_space: 使用するパラメータ空間。
            generated_param_space: LLM が自動生成したパラメータ空間。レポートに記載される。

        Returns:
            構築済みの Supervisor。
        """
        llm_provider = self._resolve_llm_provider()
        supervisor_llm = llm_provider.get_llm(temperature=0)
        expert_llm = llm_provider.get_llm(temperature=0.3)
        seed = self._config.seed

        expert_system_prompt = build_system_prompt(
            EXPERT_AGENT_DEFAULT_PROMPT,
            self._config.prompts.get("expert_agent"),
        )
        tools: list[BaseTool] = [
            SobolSearchTool(
                adapter=adapter,
                param_space=param_space,
                seed=seed,
                name="sobol_search",
                description="Sobol 列による準ランダム探索",
            ),
            BayesianOptimizationTool(
                adapter=adapter,
                param_space=param_space,
                seed=seed,
                name="bayesian_optimization",
                description="Optuna を用いたベイズ最適化による探索",
            ),
            ExpertAgentTool(
                adapter=adapter,
                param_space=param_space,
                llm=expert_llm,
                system_prompt=expert_system_prompt,
                name="expert_agent",
                description="専門家 AI エージェントによる決め打ち探索",
            ),
            NarrowSearchSpaceTool(
                param_space=param_space,
                name="narrow_search_space",
                description=(
                    "過去の探索結果をもとに探索空間を狭める。"
                    "param_updates に JSON 文字列で新しい範囲を指定する。"
                    '例: [{"name": "learning_rate", "low": 0.05, "high": 0.1}]'
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
