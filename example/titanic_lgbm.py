"""Titanic データセットを使った LightGBM ハイパーパラメーター最適化の使用例。

このスクリプトは HPO-Agent の基本的な使い方を示す。
Titanic の生存予測タスクで LGBMClassifier のハイパーパラメーターを最適化する。

前提条件:
    - .env ファイルに LLM_API_KEY と LLM_MODEL_NAME を設定済みであること
    - `uv sync` で依存関係がインストール済みであること

実行方法:
    uv run python example/titanic_lgbm.py
"""

from __future__ import annotations

import pathlib

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hpo_agent import HPOAgent

# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "Titanic-Dataset.csv"


def load_titanic(path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    """Titanic CSV を読み込み、特徴量とターゲットを返す。

    欠損値処理・カテゴリカルエンコーディングを行う。

    Args:
        path: CSV ファイルのパス。

    Returns:
        (X, y) のタプル。X は特徴量 DataFrame、y は生存フラグ Series。
    """
    df = pd.read_csv(path)

    # 目的変数
    y = df["Survived"]

    # 使用する特徴量（高カーディナリティ列・ID 列は除外）
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features].copy()

    # 欠損値補完
    X["Age"] = X["Age"].fillna(X["Age"].median())
    X["Fare"] = X["Fare"].fillna(X["Fare"].median())
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

    # カテゴリカルエンコーディング
    for col in ["Sex", "Embarked"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y


# ---------------------------------------------------------------------------
# 評価関数
# ---------------------------------------------------------------------------


def eval_fn(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """ホールドアウト精度を返す評価関数。

    Args:
        model: 学習済み LGBMClassifier。
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
    """HPO-Agent を使って Titanic の LightGBM モデルを最適化する。"""
    print("=== Titanic LightGBM HPO-Agent Example ===\n")

    # データ読み込み
    X, y = load_titanic(DATA_PATH)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples\n")

    # ベースモデル
    model = lgb.LGBMClassifier(verbosity=-1, n_estimators=100)

    # HPOAgent 実行
    agent = HPOAgent(
        model=model,
        eval_fn=eval_fn,
        n_trials=20,
        X=X_train,
        y=y_train,
        seed=42,
    )

    print("最適化を開始します（n_trials=20）...\n")
    result = agent.run()

    # 結果表示
    print("\n=== 最適化結果 ===")
    print(f"最良スコア（学習データ）: {result.best_score:.4f}")
    print(f"最良パラメータ: {result.best_params}\n")

    # 最良パラメータで再学習してバリデーション精度を確認
    best_model = lgb.LGBMClassifier(
        verbosity=-1, n_estimators=100, **result.best_params
    )
    best_model.fit(X_train, y_train)
    val_acc = accuracy_score(y_val, best_model.predict(X_val))
    print(f"バリデーション精度: {val_acc:.4f}\n")

    # 試行履歴
    print("=== 試行履歴（先頭5件）===")
    print(result.trials_df.head())

    # Markdown レポート
    print("\n=== レポート ===")
    print(result.report)


if __name__ == "__main__":
    main()
