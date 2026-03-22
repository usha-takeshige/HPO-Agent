"""sklearn RandomForestClassifier を使ったハイパーパラメーター最適化の使用例。

このスクリプトは HPO-Agent の scikit-learn モデル対応を示す最小使用例。
make_classification で生成した合成データで RandomForestClassifier を最適化する。

前提条件:
    - .env ファイルに LLM_API_KEY と LLM_MODEL_NAME を設定済みであること
    - `uv sync` で依存関係がインストール済みであること

実行方法:
    uv run python example/sklearn_rf_example.py
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from hpo_agent import HPOAgent, ParamSpace, ParamSpec

# ---------------------------------------------------------------------------
# パラメータ空間（sklearn は param_space の指定が必須）
# ---------------------------------------------------------------------------

PARAM_SPACE = ParamSpace(
    specs=(
        ParamSpec(name="n_estimators", type="int", low=50, high=500),
        ParamSpec(name="max_depth", type="int", low=3, high=20),
        ParamSpec(name="min_samples_split", type="int", low=2, high=20),
        ParamSpec(name="min_samples_leaf", type="int", low=1, high=10),
        ParamSpec(name="max_features", type="categorical", choices=("sqrt", "log2")),
    )
)


# ---------------------------------------------------------------------------
# 評価関数
# ---------------------------------------------------------------------------


def eval_fn(model: RandomForestClassifier, X: Any, y: Any) -> float:
    """ホールドアウト精度を返す評価関数。

    Args:
        model: 学習済み RandomForestClassifier。
        X: 特徴量。
        y: ターゲット。

    Returns:
        accuracy_score（0.0〜1.0、高いほど良い）。
    """
    return float(accuracy_score(y, model.predict(X)))


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    """HPO-Agent を使って RandomForestClassifier を最適化する。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== sklearn RandomForestClassifier HPO-Agent Example ===\n")

    # 合成データ生成
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples\n")

    # ベースモデル
    model = RandomForestClassifier(random_state=42)

    # HPOAgent 実行（sklearn モデルは param_space の指定が必須）
    agent = HPOAgent(
        model=model,
        eval_fn=eval_fn,
        n_trials=20,
        X=X_train,
        y=y_train,
        param_space=PARAM_SPACE,
        seed=42,
    )

    print("最適化を開始します（n_trials=20）...\n")
    result = agent.run()

    # 結果表示
    print("\n=== 最適化結果 ===")
    print(f"最良スコア（学習データ）: {result.best_score:.4f}")
    print(f"最良パラメータ: {result.best_params}\n")

    # 最良パラメータで再学習してバリデーション精度を確認
    best_model = RandomForestClassifier(random_state=42, **result.best_params)
    best_model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, best_model.predict(X_val))
    print(f"バリデーション精度: {val_acc:.4f}\n")

    # 試行履歴 CSV 出力
    output_dir = pathlib.Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "sklearn_rf_trials.csv"
    result.trials_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"試行履歴を保存しました: {csv_path}\n")

    # 試行履歴
    print("=== 試行履歴（先頭5件）===")
    print(result.trials_df.head())

    # Markdown レポート
    print("\n=== レポート ===")
    print(result.report)
    report_path = output_dir / "sklearn_rf_hpo_report.md"
    with open(report_path, "w") as f:
        f.write(result.report)


if __name__ == "__main__":
    main()
