"""Model adapter abstractions and implementations."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import lightgbm as lgb

from hpo_agent.models import ParamSpace, ParamSpec


class ModelAdapterBase(ABC):
    """モデルのパラメータ空間取得・評価実行のインターフェースを定義する。

    具象クラスは特定の ML フレームワーク（LightGBM, sklearn, PyTorch 等）に対して
    このインターフェースを実装することで、Supervisor からの差し替えを可能にする。
    """

    @abstractmethod
    def get_default_param_space(self) -> ParamSpace:
        """モデル固有のデフォルトパラメータ空間を返す。"""
        ...

    @abstractmethod
    def evaluate(self, params: dict[str, Any]) -> float:
        """指定したパラメータでモデルを評価し、スコアを返す。"""
        ...


_LIGHTGBM_DEFAULT_PARAM_SPACE = ParamSpace(
    specs=(
        ParamSpec(name="num_leaves", type="int", low=20, high=300),
        ParamSpec(name="max_depth", type="int", low=3, high=12),
        ParamSpec(name="learning_rate", type="float", low=1e-4, high=0.3, log=True),
        ParamSpec(name="n_estimators", type="int", low=50, high=1000),
        ParamSpec(name="subsample", type="float", low=0.5, high=1.0),
        ParamSpec(name="colsample_bytree", type="float", low=0.5, high=1.0),
        ParamSpec(name="reg_alpha", type="float", low=1e-8, high=10.0, log=True),
        ParamSpec(name="reg_lambda", type="float", low=1e-8, high=10.0, log=True),
    )
)


class LightGBMAdapter(ModelAdapterBase):
    """LightGBM に対してパラメータ空間取得・評価を実行するアダプター。

    evaluate() は毎回 deepcopy でモデルを複製するため、元のモデルオブジェクトは変更されない。

    Args:
        model: チューニング対象の LightGBM モデル。
        eval_fn: 評価関数。シグネチャ: (model, X, y) -> float。大きいほど良いスコア。
        X: 特徴量データ。
        y: ターゲットデータ。
    """

    def __init__(
        self,
        model: lgb.LGBMModel,
        eval_fn: Callable[..., float],
        X: Any,
        y: Any,
    ) -> None:
        """LightGBMAdapter を初期化する。"""
        self._model = model
        self._eval_fn = eval_fn
        self._X = X
        self._y = y

    def get_default_param_space(self) -> ParamSpace:
        """LightGBM のデフォルトパラメータ空間（8パラメータ）を返す。"""
        return _LIGHTGBM_DEFAULT_PARAM_SPACE

    def evaluate(self, params: dict[str, Any]) -> float:
        """パラメータを設定してモデルを学習・評価し、スコアを返す。

        元のモデルオブジェクトは変更しない（deepcopy を使用）。
        """
        model_copy = copy.deepcopy(self._model)
        model_copy.set_params(**params)
        model_copy.fit(self._X, self._y)
        return self._eval_fn(model_copy, self._X, self._y)
