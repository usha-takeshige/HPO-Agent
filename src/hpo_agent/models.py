"""Data classes for HPO-Agent.

このモジュールはエージェント全体で使用する不変・可変データクラスを定義する。
振る舞いを持つクラスとデータ保持クラスを明確に分離するため、
ロジックはここに置かない（`__post_init__` によるバリデーションを除く）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class ParamSpec:
    """1つのハイパーパラメータの型・範囲・スケールを保持する。

    Attributes:
        name: パラメータ名。
        type: パラメータの型。"int" / "float" / "categorical" のいずれか。
        low: 数値型の下限。
        high: 数値型の上限。
        choices: カテゴリカル型の選択肢。
        log: 対数スケールフラグ。True の場合は low > 0 が必要。
    """

    name: str
    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    choices: tuple[str | int | float, ...] | None = None
    log: bool = False

    def __post_init__(self) -> None:
        """バリデーション: log=True の場合 low > 0 でなければならない."""
        if self.log and self.type in ("int", "float"):
            if self.low is None or self.low <= 0:
                raise ValueError(
                    f"ParamSpec '{self.name}': log=True の場合 low > 0 が必要です。"
                    f" low={self.low}"
                )


@dataclass(frozen=True)
class ParamSpace:
    """パラメータ仕様の集合を保持する。

    Attributes:
        specs: ParamSpec のタプル。
    """

    specs: tuple[ParamSpec, ...]


class ParamSpecSchema(BaseModel):
    """LLM の structured output 用：1つのハイパーパラメータ仕様スキーマ。

    ParamSpec の Pydantic 版。choices は tuple ではなく list で受け取り、
    to_param_spec() で frozen dataclass の ParamSpec に変換する。
    """

    name: str = Field(description="パラメータ名")
    type: Literal["int", "float", "categorical"] = Field(
        description='パラメータの型。"int" / "float" / "categorical" のいずれか'
    )
    low: float | None = Field(default=None, description="数値型の下限")
    high: float | None = Field(default=None, description="数値型の上限")
    choices: list[str | int | float] | None = Field(
        default=None, description="カテゴリカル型の選択肢リスト"
    )
    log: bool = Field(default=False, description="対数スケールフラグ")

    def to_param_spec(self) -> ParamSpec:
        """ParamSpec frozen dataclass に変換する。"""
        return ParamSpec(
            name=self.name,
            type=self.type,
            low=self.low,
            high=self.high,
            choices=tuple(self.choices) if self.choices is not None else None,
            log=self.log,
        )


class ParamSpaceSchema(BaseModel):
    """LLM の structured output 用：パラメータ空間スキーマ。

    ParamSpace の Pydantic 版。to_param_space() で frozen dataclass に変換する。
    """

    specs: list[ParamSpecSchema] = Field(description="ハイパーパラメータ仕様のリスト")

    def to_param_space(self) -> ParamSpace:
        """ParamSpace frozen dataclass に変換する。"""
        return ParamSpace(specs=tuple(s.to_param_spec() for s in self.specs))


@dataclass
class TrialRecord:
    """1回の試行結果を保持する。

    Attributes:
        trial_id: 試行 ID（0 始まりの連番）。
        params: 試行したパラメータの辞書。
        score: 評価スコア。
        tool_used: 使用したツール名。
        timestamp: 試行開始日時。
        eval_duration: モデルの学習・評価にかかった時間（秒）。
        algo_duration: アルゴリズムが次の実験点を算出するのにかかった時間（秒）。
        reasoning: AI の判断理由。
    """

    trial_id: int
    params: dict[str, Any]
    score: float
    tool_used: str
    timestamp: datetime
    eval_duration: float = 0.0
    algo_duration: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """TrialRecord を辞書に変換する。timestamp は ISO 文字列にシリアライズする。"""
        return {
            "trial_id": self.trial_id,
            **self.params,
            "score": self.score,
            "tool_used": self.tool_used,
            "timestamp": self.timestamp.isoformat(),
            "eval_duration": self.eval_duration,
            "algo_duration": self.algo_duration,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class HPOConfig:
    """エージェント実行に必要な設定値を保持する。

    Attributes:
        model: チューニング対象モデル。
        eval_fn: ユーザー定義評価関数。シグネチャ: (model, X, y) -> float。
        n_trials: 総試行回数。
        X: 特徴量データ。
        y: ターゲットデータ。
        param_space: 最適化対象パラメータ空間。None の場合はアダプターのデフォルトを使用。
        seed: 乱数シード。None の場合は非決定的。
        prompts: エージェント別追加プロンプトの辞書。キー例: "supervisor", "expert_agent"。
        llm_model: LLM モデル名（.env 設定の上書き用）。
    """

    model: Any
    eval_fn: Callable[..., float]
    n_trials: int
    X: Any = None
    y: Any = None
    param_space: ParamSpace | None = None
    seed: int | None = None
    prompts: dict[str, str] = field(default_factory=dict)
    llm_model: str | None = None


@dataclass
class HPOResult:
    """最終出力（最良パラメータ・スコア・履歴・レポート）を保持する。

    Attributes:
        best_params: 最良スコアを達成したパラメータの辞書。
        best_score: 最良スコア。
        trials_df: 全試行履歴の DataFrame。
        report: Markdown 形式のレポート文字列。
    """

    best_params: dict[str, Any]
    best_score: float
    trials_df: pd.DataFrame
    report: str
