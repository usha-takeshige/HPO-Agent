"""HPO tool abstractions and implementations."""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from datetime import datetime
from math import exp, log
from typing import Any

import numpy as np
import optuna
from langchain_core.tools import BaseTool
from pydantic import ConfigDict, Field
from scipy.stats.qmc import Sobol  # type: ignore[import-untyped]

from hpo_agent.adapters import ModelAdapterBase
from hpo_agent.models import ParamSpace, TrialRecord

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HPOToolBase(BaseTool):
    """HPO ツール（探索アルゴリズム）のインターフェースを定義する。

    LangChain の BaseTool を継承し、Supervisor が bind_tools() で利用できるようにする。
    具象クラスは _run() を実装して特定の探索アルゴリズムを提供する。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: ModelAdapterBase = Field(...)
    param_space: ParamSpace = Field(...)

    @abstractmethod
    def _run(
        self,
        n_trials: int,
        trial_history: list[TrialRecord],
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """探索を実行して TrialRecord のリストを返す。

        Args:
            n_trials: 実行する試行回数。
            trial_history: これまでの試行履歴。ウォームスタートに使用。

        Returns:
            新たに実施した試行の TrialRecord リスト。
        """
        ...


# ---------------------------------------------------------------------------
# SobolSearchTool
# ---------------------------------------------------------------------------


class SobolSearchTool(HPOToolBase):
    """Sobol 列による準ランダム探索ツール。

    scipy の Sobol 列を使って数値パラメータを均一にサンプリングし、
    カテゴリカルパラメータは numpy のランダム選択で処理する。
    同じ seed を与えると再現可能なパラメータ列が生成される。
    """

    seed: int | None = None

    def _run(
        self,
        n_trials: int,
        trial_history: list[TrialRecord],
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """Sobol 列でパラメータをサンプリングして評価する。"""
        numerical_specs = [
            s for s in self.param_space.specs if s.type in ("int", "float")
        ]
        categorical_specs = [
            s for s in self.param_space.specs if s.type == "categorical"
        ]

        # --- Sobol サンプリング（数値パラメータ）---
        algo_start = time.perf_counter()
        d = len(numerical_specs)
        if d > 0:
            sampler = Sobol(d=d, scramble=True, seed=self.seed)
            samples = sampler.random(n_trials)  # shape: (n_trials, d)
        else:
            samples = np.zeros((n_trials, 0))

        # カテゴリカルパラメータのランダム選択
        rng = np.random.default_rng(seed=self.seed)
        algo_duration = time.perf_counter() - algo_start

        records: list[TrialRecord] = []
        for i in range(n_trials):
            params: dict[str, Any] = {}

            # 数値パラメータのマッピング
            for j, spec in enumerate(numerical_specs):
                u = float(samples[i, j])
                assert spec.low is not None and spec.high is not None
                if spec.log:
                    val = exp(log(spec.low) + u * (log(spec.high) - log(spec.low)))
                else:
                    val = spec.low + u * (spec.high - spec.low)
                if spec.type == "int":
                    val = int(round(val))
                    val = max(int(spec.low), min(int(spec.high), val))
                params[spec.name] = val

            # カテゴリカルパラメータのマッピング
            for spec in categorical_specs:
                assert spec.choices is not None
                params[spec.name] = rng.choice(list(spec.choices))

            eval_start = time.perf_counter()
            score = self.adapter.evaluate(params)
            eval_duration = time.perf_counter() - eval_start

            records.append(
                TrialRecord(
                    trial_id=i,
                    params=params,
                    score=score,
                    tool_used="sobol_search",
                    timestamp=datetime.now(),
                    eval_duration=eval_duration,
                    algo_duration=algo_duration,
                )
            )

        return records


# ---------------------------------------------------------------------------
# BayesianOptimizationTool
# ---------------------------------------------------------------------------


class BayesianOptimizationTool(HPOToolBase):
    """Optuna によるベイズ最適化ツール。

    TPESampler を使って試行履歴をもとにパラメータを提案する。
    同じ seed と同じ試行履歴を与えると再現可能なパラメータ列が生成される。
    """

    seed: int | None = None

    def _run(
        self,
        n_trials: int,
        trial_history: list[TrialRecord],
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """Bayesian 最適化でパラメータをサンプリングして評価する。"""
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # ウォームスタート: 過去の試行履歴を Optuna に登録
        distributions = self._build_distributions()
        for record in trial_history:
            params_clipped = {
                k: v for k, v in record.params.items() if k in distributions
            }
            if not params_clipped:
                continue
            frozen_trial = optuna.trial.FrozenTrial(
                number=len(study.trials),
                state=optuna.trial.TrialState.COMPLETE,
                value=record.score,
                values=None,
                datetime_start=record.timestamp,
                datetime_complete=record.timestamp,
                params=params_clipped,
                distributions=distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=len(study.trials),
            )
            study.add_trial(frozen_trial)

        records: list[TrialRecord] = []
        for i in range(n_trials):
            algo_start = time.perf_counter()
            trial = study.ask(fixed_distributions=distributions)
            suggested_params = trial.params
            algo_duration = time.perf_counter() - algo_start

            eval_start = time.perf_counter()
            score = self.adapter.evaluate(suggested_params)
            eval_duration = time.perf_counter() - eval_start

            study.tell(trial, score)

            records.append(
                TrialRecord(
                    trial_id=i,
                    params=suggested_params,
                    score=score,
                    tool_used="bayesian_optimization",
                    timestamp=datetime.now(),
                    eval_duration=eval_duration,
                    algo_duration=algo_duration,
                )
            )

        return records

    def _build_distributions(
        self,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        """ParamSpace を Optuna の Distribution に変換する。"""
        distributions: dict[str, optuna.distributions.BaseDistribution] = {}
        for spec in self.param_space.specs:
            if spec.type == "int":
                assert spec.low is not None and spec.high is not None
                distributions[spec.name] = optuna.distributions.IntDistribution(
                    low=int(spec.low),
                    high=int(spec.high),
                    log=spec.log,
                )
            elif spec.type == "float":
                assert spec.low is not None and spec.high is not None
                distributions[spec.name] = optuna.distributions.FloatDistribution(
                    low=spec.low,
                    high=spec.high,
                    log=spec.log,
                )
            elif spec.type == "categorical":
                assert spec.choices is not None
                distributions[spec.name] = optuna.distributions.CategoricalDistribution(
                    choices=spec.choices,
                )
        return distributions


# ---------------------------------------------------------------------------
# ExpertAgentTool
# ---------------------------------------------------------------------------


class ExpertAgentTool(HPOToolBase):
    """専門家 AI エージェントによる決め打ち探索ツール。

    LLM に試行履歴とパラメータ空間を渡し、次のパラメータ提案を JSON で受け取る。
    JSON の解析に失敗した場合は最大 3 回まで再試行する。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: Any = Field(...)
    system_prompt: str = Field(...)

    _MAX_RETRIES: int = 3
    _MAX_HISTORY_TOP: int = 20
    _MAX_HISTORY_RECENT: int = 10

    def _run(
        self,
        n_trials: int,
        trial_history: list[TrialRecord],
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """LLM からパラメータ提案を受け取って評価する。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        selected_history = self._select_history(trial_history)
        history_json = json.dumps(
            [r.to_dict() for r in selected_history],
            ensure_ascii=False,
        )
        param_space_desc = self._build_param_space_description()
        user_message = self._build_user_message(history_json, param_space_desc)

        records: list[TrialRecord] = []
        for i in range(n_trials):
            algo_start = time.perf_counter()
            parsed: dict[str, Any] | None = None
            for attempt in range(self._MAX_RETRIES):
                response = self.llm.invoke(
                    [
                        SystemMessage(content=self.system_prompt),
                        HumanMessage(content=user_message),
                    ]
                )
                try:
                    content = response.content
                    if isinstance(content, list):
                        content = "".join(
                            b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in content
                        )
                    parsed = json.loads(content)
                    break
                except (json.JSONDecodeError, ValueError, TypeError):
                    if attempt == self._MAX_RETRIES - 1:
                        raise RuntimeError(
                            f"ExpertAgentTool: {self._MAX_RETRIES} 回連続で JSON の解析に失敗しました。"
                        )

            algo_duration = time.perf_counter() - algo_start

            assert parsed is not None
            params: dict[str, Any] = parsed.get("params", {})
            reasoning: str = parsed.get("reasoning", "")

            eval_start = time.perf_counter()
            score = self.adapter.evaluate(params)
            eval_duration = time.perf_counter() - eval_start

            records.append(
                TrialRecord(
                    trial_id=i,
                    params=params,
                    score=score,
                    tool_used="expert_agent",
                    timestamp=datetime.now(),
                    eval_duration=eval_duration,
                    algo_duration=algo_duration,
                    reasoning=reasoning,
                )
            )

        return records

    def _select_history(self, trial_history: list[TrialRecord]) -> list[TrialRecord]:
        """スコア上位 20 件 ∪ 直近 10 件を重複排除して返す。"""
        top_by_score = sorted(trial_history, key=lambda r: r.score, reverse=True)[
            : self._MAX_HISTORY_TOP
        ]
        recent = sorted(trial_history, key=lambda r: r.trial_id, reverse=True)[
            : self._MAX_HISTORY_RECENT
        ]
        seen_ids: set[int] = set()
        merged: list[TrialRecord] = []
        for r in top_by_score + recent:
            if r.trial_id not in seen_ids:
                seen_ids.add(r.trial_id)
                merged.append(r)
        return sorted(merged, key=lambda r: r.trial_id)

    def _build_param_space_description(self) -> str:
        """ParamSpace をテキスト形式に変換する。"""
        lines: list[str] = []
        for spec in self.param_space.specs:
            if spec.type == "categorical":
                assert spec.choices is not None
                lines.append(
                    f"- {spec.name}: categorical, choices={list(spec.choices)}"
                )
            else:
                scale = "log" if spec.log else "linear"
                lines.append(
                    f"- {spec.name}: {spec.type}, low={spec.low}, high={spec.high}, scale={scale}"
                )
        return "\n".join(lines)

    def _build_user_message(self, history_json: str, param_space_desc: str) -> str:
        """LLM へのユーザーメッセージを構築する。"""
        return (
            f"## パラメータ空間\n{param_space_desc}\n\n"
            f"## 試行履歴\n```json\n{history_json}\n```\n\n"
            "上記の情報を参考に、次の試行パラメータを JSON で提案してください。\n"
            '{"reasoning": "提案理由", "params": {"param_name": value, ...}}'
        )
