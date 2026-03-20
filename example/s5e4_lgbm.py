"""Spotify Podcast Listening Time データセットを使った LightGBM ハイパーパラメーター最適化の使用例。

このスクリプトは HPO-Agent の基本的な使い方を示す。
Kaggle Playground Series S5E4 の Podcast 聴取時間予測タスクで
LGBMRegressor のハイパーパラメーターを最適化する。

評価指標:
    KFold 5 分割クロスバリデーションによる RMSE（最小化）

チューニング対象パラメータ:
    num_leaves, max_depth, learning_rate, n_estimators

前提条件:
    - .env ファイルに LLM_API_KEY と LLM_MODEL_NAME を設定済みであること
    - `uv sync` で依存関係がインストール済みであること
    - data/s5e4_train.csv が存在すること

実行方法:
    uv run python example/s5e4_lgbm.py
"""

from __future__ import annotations

import copy
import logging
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

from hpo_agent import HPOAgent, ParamSpace, ParamSpec

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "s5e4_train.csv"
TARGET_COL = "Listening_Time_minutes"
IGNORE_COLS = ["id", TARGET_COL]
N_FOLDS = 5
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# データ準備
# ---------------------------------------------------------------------------


def load_data(path: pathlib.Path) -> tuple[pd.DataFrame, pd.Series]:
    """s5e4_train.csv を読み込み、特徴量とターゲットを返す。

    欠損値補完・カテゴリカルエンコーディングを行う。

    Args:
        path: CSV ファイルのパス。

    Returns:
        (X, y) のタプル。X は特徴量 DataFrame、y は聴取時間 Series。
    """
    df = pd.read_csv(path)

    y = df[TARGET_COL]
    X = df.drop(columns=IGNORE_COLS).copy()

    # 欠損値補完
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            # 数値: 中央値で補完
            X[col] = X[col].fillna(X[col].median())
        else:
            # カテゴリカル: 最頻値で補完
            X[col] = X[col].fillna(X[col].mode()[0])

    # カテゴリカルエンコーディング（ラベルエンコーディング）
    for col in X.select_dtypes(exclude="number").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    return X, y


# ---------------------------------------------------------------------------
# 評価関数（KFold CV による負の RMSE）
# ---------------------------------------------------------------------------


def eval_fn(model: lgb.LGBMRegressor, X: pd.DataFrame, y: pd.Series) -> float:
    """KFold 5 分割 CV の平均 RMSE を負値で返す評価関数。

    HPO-Agent はスコアを最大化するため、RMSE を負値に変換して返す。
    model はパラメータ設定済みのインスタンスとして渡される。
    各フォールドで deepcopy して再学習する。

    Args:
        model: パラメータ設定済みの LGBMRegressor（フィット済みだが再利用しない）。
        X: 特徴量。
        y: ターゲット。

    Returns:
        負の平均 RMSE（0 に近いほど良い）。
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores: list[float] = []

    X_arr = X.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)

    for train_idx, val_idx in kf.split(X_arr):
        X_tr, X_val = X_arr.iloc[train_idx], X_arr.iloc[val_idx]
        y_tr, y_val = y_arr.iloc[train_idx], y_arr.iloc[val_idx]

        fold_model = copy.deepcopy(model)
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_val)
        rmse_scores.append(root_mean_squared_error(y_val, preds))

    return -float(np.mean(rmse_scores))


# ---------------------------------------------------------------------------
# パラメータ空間
# ---------------------------------------------------------------------------

PARAM_SPACE = ParamSpace(
    specs=(
        ParamSpec(name="num_leaves", type="int", low=20, high=300),
        ParamSpec(name="max_depth", type="int", low=3, high=12),
        ParamSpec(name="learning_rate", type="float", low=1e-3, high=0.3, log=True),
        ParamSpec(name="n_estimators", type="int", low=100, high=1000),
    )
)

# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------


def main() -> None:
    """HPO-Agent を使って Spotify Podcast データの LightGBM モデルを最適化する。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=== Spotify Podcast Listening Time LightGBM HPO-Agent Example ===\n")

    # データ読み込み
    X, y = load_data(DATA_PATH)
    print(f"データ形状: {X.shape}, ターゲット: {TARGET_COL}\n")

    # ベースモデル
    model = lgb.LGBMRegressor(verbosity=-1, random_state=RANDOM_STATE)

    # HPOAgent 実行
    agent = HPOAgent(
        model=model,
        eval_fn=eval_fn,
        n_trials=20,
        X=X,
        y=y,
        param_space=PARAM_SPACE,
        seed=RANDOM_STATE,
    )

    print("最適化を開始します（n_trials=20, 評価指標: 5-Fold CV RMSE）...\n")
    result = agent.run()

    # 結果表示
    best_rmse = -result.best_score
    print("\n=== 最適化結果 ===")
    print(f"最良 CV RMSE: {best_rmse:.4f}")
    print(f"最良パラメータ: {result.best_params}\n")

    # 試行履歴
    print("=== 試行履歴（先頭 5 件）===")
    display_df = result.trials_df.copy()
    if "score" in display_df.columns:
        display_df["cv_rmse"] = -display_df["score"]
    print(display_df.head())

    # Markdown レポート
    print("\n=== レポート ===")
    print(result.report)


if __name__ == "__main__":
    main()
