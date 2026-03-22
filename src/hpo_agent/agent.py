"""Public entry point for HPO-Agent."""

from __future__ import annotations

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
    SUPERVISOR_DEFAULT_PROMPT,
    build_system_prompt,
)
from hpo_agent.providers import GoogleLLMProvider
from hpo_agent.report import ReportGenerator
from hpo_agent.supervisor import Supervisor
from hpo_agent.tools import (
    BayesianOptimizationTool,
    ExpertAgentTool,
    NarrowSearchSpaceTool,
    SobolSearchTool,
)


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
        X, y は不要。param_space の指定が必須。

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
        param_space: 最適化対象パラメータ空間。PyTorch では必須。
            LightGBM で None の場合はアダプターのデフォルトを使用。
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

        Returns:
            最適化結果（HPOResult）。
        """
        adapter, param_space = self._resolve_adapter()
        supervisor = self._build_supervisor(adapter, param_space)
        return supervisor.run(self._config)

    def _resolve_adapter(self) -> tuple[ModelAdapterBase, ParamSpace]:
        """モデル型に応じたアダプターと使用するパラメータ空間を返す。

        Returns:
            (adapter, param_space) のタプル。

        Raises:
            TypeError: 未対応のモデル型の場合。
            ValueError: scikit-learn / PyTorch モデルで param_space が未指定の場合。
        """
        import lightgbm as lgb
        from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

        model = self._config.model
        adapter: ModelAdapterBase
        if isinstance(model, lgb.LGBMModel):
            adapter: ModelAdapterBase = LightGBMAdapter(
                model=model,
                eval_fn=self._config.eval_fn,
                X=self._config.X,
                y=self._config.y,
            )
        elif isinstance(model, BaseEstimator):
            if self._config.param_space is None:
                raise ValueError(
                    "scikit-learn モデルを使用する場合は param_space の指定が必須です。"
                )
            adapter = SklearnAdapter(
                model=model,
                eval_fn=self._config.eval_fn,
                X=self._config.X,
                y=self._config.y,
            )
        elif callable(model):
            if self._config.param_space is None:
                raise ValueError(
                    "PyTorch モデルを使用する場合は param_space の指定が必須です。"
                )
            adapter = PyTorchAdapter(
                model_fn=model,
                eval_fn=self._config.eval_fn,
                param_space=self._config.param_space,
            )
        else:
            raise TypeError(
                f"Unsupported model type: {type(model)}. "
                "Supported: LightGBM (lgb.LGBMModel), scikit-learn (BaseEstimator subclasses), "
                "PyTorch (callable な model_fn)。"
            )

        param_space = self._config.param_space or adapter.get_default_param_space()
        return adapter, param_space

    def _resolve_llm_provider(self) -> GoogleLLMProvider:
        """環境変数から LLM プロバイダーを構築して返す。"""
        load_dotenv()
        api_key = os.environ["LLM_API_KEY"]
        model_name = self._config.llm_model or os.environ["LLM_MODEL_NAME"]
        return GoogleLLMProvider(api_key=api_key, model_name=model_name)

    def _build_supervisor(
        self,
        adapter: ModelAdapterBase,
        param_space: ParamSpace,
    ) -> Supervisor:
        """Supervisor インスタンスを構築して返す。

        Args:
            adapter: モデルアダプター。
            param_space: 使用するパラメータ空間。

        Returns:
            構築済みの Supervisor。
        """
        llm_provider = self._resolve_llm_provider()
        llm = llm_provider.get_llm()
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
                llm=llm,
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
            llm=llm,
            tools=tools,
            report_generator=ReportGenerator(),
            system_prompt=supervisor_prompt,
        )
