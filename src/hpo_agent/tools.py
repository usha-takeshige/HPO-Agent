"""HPO tool abstractions and implementations."""

from __future__ import annotations

import json
import logging
import re
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
from hpo_agent.models import ParamSpace, ParamSpec, TrialRecord

logger = logging.getLogger(__name__)

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
        effective_param_space: ParamSpace | None = None,
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """探索を実行して TrialRecord のリストを返す。

        Args:
            n_trials: 実行する試行回数。
            trial_history: これまでの試行履歴。ウォームスタートに使用。
            effective_param_space: 有効な探索空間。None の場合は self.param_space を使用する。

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
        effective_param_space: ParamSpace | None = None,
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """Sobol 列でパラメータをサンプリングして評価する。"""
        param_space = effective_param_space or self.param_space
        numerical_specs = [s for s in param_space.specs if s.type in ("int", "float")]
        categorical_specs = [s for s in param_space.specs if s.type == "categorical"]

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
            logger.info(
                "[sobol_search] trial=%d | score=%.6f | eval=%.2fs | params=%s",
                i,
                score,
                eval_duration,
                params,
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
        effective_param_space: ParamSpace | None = None,
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """Bayesian 最適化でパラメータをサンプリングして評価する。"""
        param_space = effective_param_space or self.param_space
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # ウォームスタート: 過去の試行履歴を Optuna に登録
        distributions = self._build_distributions(param_space)
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
            logger.info(
                "[bayesian_optimization] trial=%d | score=%.6f | eval=%.2fs | params=%s",
                i,
                score,
                eval_duration,
                suggested_params,
            )

        return records

    def _build_distributions(
        self,
        param_space: ParamSpace | None = None,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        """ParamSpace を Optuna の Distribution に変換する。"""
        target = param_space or self.param_space
        distributions: dict[str, optuna.distributions.BaseDistribution] = {}
        for spec in target.specs:
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
        effective_param_space: ParamSpace | None = None,
        **kwargs: Any,
    ) -> list[TrialRecord]:
        """LLM からパラメータ提案を受け取って評価する。"""
        from langchain_core.messages import HumanMessage, SystemMessage

        param_space = effective_param_space or self.param_space
        param_space_desc = self._build_param_space_description(param_space)
        # ループ内で更新するため mutable なリストとして保持
        current_history = list(trial_history)

        records: list[TrialRecord] = []
        for i in range(n_trials):
            # 試行ごとに最新の履歴でメッセージを再構築
            selected_history = self._select_history(current_history)
            history_json = json.dumps(
                [r.to_dict() for r in selected_history],
                ensure_ascii=False,
            )
            user_message = self._build_user_message(history_json, param_space_desc)

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
                    # マークダウンコードブロックを除去（Gemini が ```json ... ``` で囲む場合）
                    content = re.sub(
                        r"^```(?:json)?\s*|\s*```$", "", content.strip()
                    ).strip()
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

            new_record = TrialRecord(
                trial_id=i,
                params=params,
                score=score,
                tool_used="expert_agent",
                timestamp=datetime.now(),
                eval_duration=eval_duration,
                algo_duration=algo_duration,
                reasoning=reasoning,
            )
            records.append(new_record)
            current_history.append(new_record)
            logger.info(
                "[expert_agent] trial=%d | score=%.6f | eval=%.2fs | params=%s",
                i,
                score,
                eval_duration,
                params,
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

    def _build_param_space_description(
        self, param_space: ParamSpace | None = None
    ) -> str:
        """ParamSpace をテキスト形式に変換する。"""
        target = param_space or self.param_space
        lines: list[str] = []
        for spec in target.specs:
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


# ---------------------------------------------------------------------------
# NarrowSearchSpaceTool
# ---------------------------------------------------------------------------


class NarrowSearchSpaceTool(BaseTool):
    """過去の探索結果をもとに探索空間を狭めるツール。

    スーパーバイザーの LLM がこのツールを呼び出すと、SupervisorState の
    current_param_space が更新され、以降の探索ツール（SobolSearchTool,
    BayesianOptimizationTool, ExpertAgentTool）は狭めた空間を使用する。

    入力 (param_updates) は JSON 配列文字列で、各要素は更新するパラメータの仕様を表す。
    数値型の例: [{"name": "learning_rate", "low": 0.05, "high": 0.1}]
    カテゴリカル型の例: [{"name": "boosting_type", "choices": ["gbdt"]}]

    検証ルール:
        - name が元の param_space に存在すること。
        - 数値型: new_low >= original.low かつ new_high <= original.high かつ new_low < new_high。
        - categorical 型: new_choices が original.choices の部分集合であること。

    更新対象外のパラメータは元の仕様を引き継ぐ。
    エラー発生時は "Error: ..." で始まる文字列を返す。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    param_space: ParamSpace = Field(...)

    def _run(self, param_updates: str, **kwargs: Any) -> str:
        """param_updates を検証し、新しい param_space の説明文字列を返す。

        Args:
            param_updates: JSON 配列文字列。各パラメータの新しい範囲を指定する。

        Returns:
            成功時: 新しい param_space の説明文字列。
            失敗時: "Error: ..." で始まるエラーメッセージ文字列。
        """
        result = self._build_narrowed_space(param_updates)
        if isinstance(result, str):
            return result
        return self._describe_param_space(result)

    def _build_narrowed_space(self, param_updates: str) -> ParamSpace | str:
        """param_updates を解析・検証して新しい ParamSpace を返す。

        Args:
            param_updates: JSON 配列文字列。

        Returns:
            成功時: 新しい ParamSpace。失敗時: "Error: ..." で始まるエラーメッセージ文字列。
        """
        try:
            updates: list[dict[str, Any]] = json.loads(param_updates)
        except json.JSONDecodeError as e:
            return f"Error: JSON パースエラー: {e}"

        spec_map = {s.name: s for s in self.param_space.specs}
        updated_specs: dict[str, ParamSpec] = {}

        for update in updates:
            name = update.get("name")
            if not isinstance(name, str) or name not in spec_map:
                return f"Error: パラメータ '{name}' は元の param_space に存在しません。"

            original = spec_map[name]

            if original.type in ("int", "float"):
                assert original.low is not None and original.high is not None
                new_low: float = float(update.get("low", original.low))
                new_high: float = float(update.get("high", original.high))
                if new_low < original.low:
                    return (
                        f"Error: '{name}' の low={new_low} は"
                        f" 元の low={original.low} より小さくできません。"
                    )
                if new_high > original.high:
                    return (
                        f"Error: '{name}' の high={new_high} は"
                        f" 元の high={original.high} より大きくできません。"
                    )
                if new_low >= new_high:
                    return (
                        f"Error: '{name}' の low={new_low} は"
                        f" high={new_high} より小さくする必要があります。"
                    )
                updated_specs[name] = ParamSpec(
                    name=name,
                    type=original.type,
                    low=new_low,
                    high=new_high,
                    log=original.log,
                )
            elif original.type == "categorical":
                assert original.choices is not None
                raw_choices = update.get("choices", list(original.choices))
                new_choices = tuple(raw_choices)
                invalid = set(new_choices) - set(original.choices)
                if invalid:
                    return f"Error: '{name}' の choices に元にない値が含まれます: {invalid}"
                updated_specs[name] = ParamSpec(
                    name=name,
                    type="categorical",
                    choices=new_choices,
                )

        # 更新対象でないパラメータは元のまま引き継ぎ、元の順序を維持する
        new_specs: list[ParamSpec] = []
        for spec in self.param_space.specs:
            new_specs.append(updated_specs.get(spec.name, spec))

        return ParamSpace(specs=tuple(new_specs))

    def _describe_param_space(self, param_space: ParamSpace) -> str:
        """ParamSpace をテキスト形式で説明する。

        Args:
            param_space: 説明対象の ParamSpace。

        Returns:
            各パラメータの仕様を1行ずつ列挙した文字列。
        """
        lines: list[str] = []
        for spec in param_space.specs:
            if spec.type == "categorical":
                assert spec.choices is not None
                lines.append(
                    f"- {spec.name}: categorical, choices={list(spec.choices)}"
                )
            else:
                scale = "log" if spec.log else "linear"
                lines.append(
                    f"- {spec.name}: {spec.type},"
                    f" low={spec.low}, high={spec.high}, scale={scale}"
                )
        return "\n".join(lines)
