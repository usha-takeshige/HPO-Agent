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
# モデルファクトリ・評価関数
# ---------------------------------------------------------------------------


def model_fn(params: dict[str, Any]) -> MLP:
    """パラメータからモデルを生成するファクトリ関数。

    Args:
        params: ハイパーパラメータの辞書。

    Returns:
        初期化済みの MLP モデル。
    """
    return MLP(hidden_size=int(params["hidden_size"]), dropout=float(params["dropout"]))


def eval_fn(model: MLP) -> float:
    """モデルを学習・評価してスコア（負の MSE）を返す評価関数。

    HPO-Agent は大きいほど良いスコアを期待するため、MSE の符号を反転して返す。

    Args:
        model: ファクトリ関数が生成した MLP モデル。

    Returns:
        負の検証 MSE（高いほど良い）。
    """
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


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    """HPO-Agent を使って MLP の hidden_size と dropout を最適化する。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== PyTorch MLP HPO-Agent Example ===\n")

    agent = HPOAgent(
        model=model_fn,
        eval_fn=eval_fn,
        n_trials=15,
        param_space=param_space,
        seed=42,
    )

    print("最適化を開始します（n_trials=15）...\n")
    result = agent.run()

    print("\n=== 最適化結果 ===")
    print(f"最良スコア（負の MSE）: {result.best_score:.4f}")
    print(f"最良パラメータ: {result.best_params}\n")

    print("=== 試行履歴（先頭5件）===")
    print(result.trials_df.head())

    print("\n=== レポート ===")
    print(result.report)


if __name__ == "__main__":
    main()
