"""テスト共有フィクスチャとモック定義。

全テストファイルで共有されるフィクスチャ・モッククラスを定義する。
LLM 呼び出しは必ずここで定義したモックに置き換え、API キー不要でテストを実行できる。
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from hpo_agent.adapters import ModelAdapterBase
from hpo_agent.models import ParamSpace, ParamSpec, TrialRecord

# ---------------------------------------------------------------------------
# DummyAdapter（全テストで共用）
# ---------------------------------------------------------------------------


class DummyAdapter(ModelAdapterBase):
    """テスト用の軽量アダプター。

    evaluate() は固定スコア 0.85 を返す。
    sleep(0.001) を挿入して eval_duration > 0 を保証する。
    """

    def __init__(self, param_space: ParamSpace) -> None:
        """DummyAdapter を初期化する。"""
        self._param_space = param_space

    def get_default_param_space(self) -> ParamSpace:
        """テスト用パラメータ空間を返す。"""
        return self._param_space

    def evaluate(self, params: dict[str, Any]) -> float:
        """固定スコア 0.85 を返す（eval_duration > 0 を保証するため sleep を挿入）。"""
        time.sleep(0.001)
        return 0.85


# ---------------------------------------------------------------------------
# Fixtures: パラメータ空間・試行履歴
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_param_space() -> ParamSpace:
    """テスト用の3パラメータ空間。"""
    return ParamSpace(
        specs=(
            ParamSpec(name="num_leaves", type="int", low=20, high=100),
            ParamSpec(name="learning_rate", type="float", low=0.01, high=0.3, log=True),
            ParamSpec(
                name="boosting_type",
                type="categorical",
                choices=("gbdt", "dart"),
            ),
        )
    )


@pytest.fixture
def sample_trial_records() -> list[TrialRecord]:
    """5件のサンプル試行履歴（スコア昇順）。"""
    return [
        TrialRecord(
            trial_id=i,
            params={"num_leaves": 20 + i * 10, "learning_rate": 0.05},
            score=0.70 + i * 0.05,
            tool_used="sobol_search",
            timestamp=datetime(2026, 3, 1, 0, i),
            eval_duration=0.5 + i * 0.1,
            algo_duration=0.01,
        )
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# Fixtures: モック LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_supervisor_llm() -> MagicMock:
    """Supervisor の LLM モック。sobol_search を n_trials=5 で1回実行して終了。"""
    llm = MagicMock()
    # 1回目：ツール呼び出し指示 → 2回目：終了（ツール呼び出しなし）
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = [
        AIMessage(
            content="sobol_search を実行します。",
            tool_calls=[
                {"name": "sobol_search", "args": {"n_trials": 5}, "id": "call_1"}
            ],
        ),
        AIMessage(content="最適化完了。", tool_calls=[]),
        MagicMock(content="AI 考察: 最適化結果を分析しました。"),  # generate_final() 用
    ]
    return llm


@pytest.fixture
def mock_expert_llm() -> MagicMock:
    """ExpertAgentTool の LLM モック。正常 JSON を返す。"""
    llm = MagicMock()
    llm.invoke.return_value.content = '{"reasoning": "テスト根拠", "params": {"num_leaves": 64, "learning_rate": 0.05}}'
    return llm


# ---------------------------------------------------------------------------
# Fixtures: アダプター
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_adapter(simple_param_space: ParamSpace) -> DummyAdapter:
    """DummyAdapter フィクスチャ（simple_param_space を使用）。"""
    return DummyAdapter(param_space=simple_param_space)


# ---------------------------------------------------------------------------
# Fixtures: 確定的評価関数
# ---------------------------------------------------------------------------


@pytest.fixture
def deterministic_eval_fn():  # type: ignore[no-untyped-def]
    """num_leaves=64 付近で最大スコアを持つ確定的な2次関数。"""

    def eval_fn(model: Any, X: Any, y: Any) -> float:
        """評価関数: -(num_leaves - 64)^2 / 10000 + 0.9."""
        num_leaves = model.get_params().get("num_leaves", 50)
        return -((num_leaves - 64) ** 2) / 10000 + 0.9

    return eval_fn


@pytest.fixture
def narrow_search_space_tool(simple_param_space: ParamSpace) -> Any:
    """NarrowSearchSpaceTool フィクスチャ（simple_param_space を使用）。"""
    from hpo_agent.tools import NarrowSearchSpaceTool

    return NarrowSearchSpaceTool(
        param_space=simple_param_space,
        name="narrow_search_space",
        description="test",
    )


# ---------------------------------------------------------------------------
# Fixtures: LightGBM セットアップ（A-2/A-3 テスト用）
# ---------------------------------------------------------------------------


@pytest.fixture
def lgbm_binary_setup():  # type: ignore[no-untyped-def]
    """軽量な LightGBM 二値分類のセットアップ。"""
    import lightgbm as lgb
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = lgb.LGBMClassifier(verbosity=-1, n_estimators=5)

    def eval_fn(m: Any, X: Any, y: Any) -> float:
        """精度を評価関数として返す。"""
        return float(accuracy_score(y, m.predict(X)))

    return model, eval_fn, X, y


# ---------------------------------------------------------------------------
# Fixtures: sklearn セットアップ（CMP-27〜29 テスト用）
# ---------------------------------------------------------------------------


@pytest.fixture
def sklearn_binary_setup():  # type: ignore[no-untyped-def]
    """軽量な sklearn RandomForestClassifier 二値分類のセットアップ。"""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = RandomForestClassifier(n_estimators=5, random_state=42)

    def eval_fn(m: Any, X: Any, y: Any) -> float:
        """精度を評価関数として返す。"""
        return float(accuracy_score(y, m.predict(X)))

    return model, eval_fn, X, y


# ---------------------------------------------------------------------------
# Fixtures: PyTorch セットアップ（CMP-30〜34 テスト用）
# ---------------------------------------------------------------------------


@pytest.fixture
def pytorch_setup() -> tuple[Any, Any, ParamSpace]:
    """実際の torch.nn.Module を使った軽量 PyTorch セットアップ。

    SimpleNet は Linear 層1つの最小構成。
    eval_fn は学習なしの forward pass でスコアを返す（テストの高速化のため）。
    """
    import torch
    import torch.nn as nn

    class SimpleNet(nn.Module):
        """テスト用の単純な全結合ネットワーク。"""

        def __init__(self, hidden_size: int) -> None:
            """SimpleNet を初期化する。"""
            super().__init__()
            self.fc = nn.Linear(10, hidden_size)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """forward パスを実行する。"""
            return self.fc(x)

    def model_fn(params: dict[str, Any]) -> nn.Module:
        """パラメータを受け取り SimpleNet を返すファクトリ関数。"""
        return SimpleNet(hidden_size=int(params["hidden_size"]))

    def eval_fn(model: nn.Module) -> float:
        """forward pass の出力絶対値平均をスコアとして返す。"""
        x = torch.randn(5, 10)
        with torch.no_grad():
            out = model(x)
        return float(out.abs().mean().item())

    param_space = ParamSpace(
        specs=(ParamSpec(name="hidden_size", type="int", low=4, high=32),)
    )
    return model_fn, eval_fn, param_space
