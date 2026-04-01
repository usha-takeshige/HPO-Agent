"""シンプルな MLP を HPO-Agent で最適化する PyTorch 使用例。

このスクリプトは HPO-Agent の PyTorch 対応の基本的な使い方を示す。
ランダム生成データを使った回帰タスクで MLP のハイパーパラメーターを最適化する。

前提条件:
    - .env ファイルに LLM_API_KEY と LLM_MODEL_NAME を設定済みであること
    - `uv sync` で依存関係がインストール済みであること
    - `torch` がインストールされていること

実行方法:
    uv run python example/simple_pytorch.py
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from hpo_agent import HPOAgent, ParamSpace, ParamSpec

# ---------------------------------------------------------------------------
# モデル定義
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    """2層の全結合ネットワーク。

    Attributes:
        net: 全結合層のシーケンシャルモジュール。
    """

    def __init__(self, hidden_size: int, dropout: float) -> None:
        """MLP を初期化する。"""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward パスを実行する。"""
        return self.net(x).squeeze(1)


# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------

torch.manual_seed(42)
X_train = torch.randn(500, 10)
y_train = (
    X_train[:, 0] * 2.0 + X_train[:, 1] - X_train[:, 2] * 0.5 + torch.randn(500) * 0.1
)

X_val = torch.randn(100, 10)
y_val = X_val[:, 0] * 2.0 + X_val[:, 1] - X_val[:, 2] * 0.5 + torch.randn(100) * 0.1


# ---------------------------------------------------------------------------
# 評価関数
# ---------------------------------------------------------------------------


def eval_fn(params: dict[str, Any]) -> float:
    """パラメータからモデルを構築・学習・評価してスコア（負の MSE）を返す評価関数。

    HPO-Agent は大きいほど良いスコアを期待するため、MSE の符号を反転して返す。

    Args:
        params: ハイパーパラメータの辞書。

    Returns:
        負の検証 MSE（高いほど良い）。
    """
    model = MLP(
        hidden_size=int(params["hidden_size"]), dropout=float(params["dropout"])
    )
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_mse = loss_fn(val_pred, y_val).item()

    return -val_mse  # 符号反転: MSE が小さいほどスコアが高い


# ---------------------------------------------------------------------------
# パラメータ空間定義
# ---------------------------------------------------------------------------

param_space = ParamSpace(
    specs=(
        ParamSpec(name="hidden_size", type="int", low=16, high=256),
        ParamSpec(name="dropout", type="float", low=0.0, high=0.5),
    )
)

# 部分指定: パラメータ名と型のみ指定し、範囲は LLM が補完する
partial_param_space = ParamSpace(
    specs=(
        ParamSpec(name="hidden_size", type="int", high=256),  # 範囲は LLM が補完
        ParamSpec(name="dropout", type="float", low=0.0),  # 範囲は LLM が補完
    )
)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    """HPO-Agent を使って MLP の hidden_size と dropout を最適化する。

    param_space を明示的に指定する方法・LLM に自動生成させる方法・
    パラメータ名のみ指定して LLM に範囲を補完させる方法の3通りを示す。
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    print("=== PyTorch MLP HPO-Agent Example ===\n")

    # --- 方法1: param_space を明示的に指定する（従来の方法）---
    # agent = HPOAgent(
    #     eval_fn=eval_fn,
    #     n_trials=15,
    #     param_space=param_space,  # 明示的に指定
    #     seed=42,
    # )

    # --- 方法2: param_space を省略して LLM に自動生成させる ---
    # eval_fn のソースコードをもとに LLM が探索空間を設計する
    # agent = HPOAgent(
    #     eval_fn=eval_fn,
    #     n_trials=15,
    #     # param_space を省略 → LLM が自動生成
    #     seed=42,
    # )

    # --- 方法3: パラメータ名と型のみ指定して LLM に範囲を補完させる ---
    # パラメータ名（hidden_size / dropout）を明示することで
    # LLM が誤った名前を生成するリスクを排除しつつ、範囲設計を LLM に委ねる
    agent = HPOAgent(
        eval_fn=eval_fn,
        n_trials=5,
        param_space=partial_param_space,  # 名前・型のみ指定 → LLM が範囲を補完
        seed=42,
    )

    print("Starting optimization (n_trials=5)...\n")
    result = agent.run()

    print("\n=== Optimization Results ===")
    print(f"Best score (negative MSE): {result.best_score:.4f}")
    print(f"Best params: {result.best_params}\n")

    # 試行履歴 CSV 出力
    output_dir = pathlib.Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "pytorch_trials.csv"
    result.trials_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Trial history saved: {csv_path}\n")

    print("=== Trial History (first 5) ===")
    print(result.trials_df.head())

    # Markdown レポート出力
    print("\n=== Report ===")
    print(result.report)
    report_path = output_dir / "pytorch_hpo_report.md"
    with open(report_path, "w") as f:
        f.write(result.report)
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
