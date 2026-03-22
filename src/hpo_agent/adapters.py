"""Model adapter abstractions and implementations."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import lightgbm as lgb
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.base import clone as sklearn_clone

from hpo_agent.models import ParamSpace


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
        """LightGBM はデフォルトパラメータ空間を提供しない。LLM が自動生成する。"""
        raise NotImplementedError(
            "LightGBMAdapter はデフォルトパラメータ空間を持ちません。"
            " param_space を省略した場合は LLM が自動生成します。"
        )

    def evaluate(self, params: dict[str, Any]) -> float:
        """パラメータを設定してモデルを学習・評価し、スコアを返す。

        元のモデルオブジェクトは変更しない（deepcopy を使用）。
        """
        model_copy = copy.deepcopy(self._model)
        model_copy.set_params(**params)
        model_copy.fit(self._X, self._y)
        return self._eval_fn(model_copy, self._X, self._y)


class SklearnAdapter(ModelAdapterBase):
    """sklearn 互換モデル（BaseEstimator）に対してパラメータ空間取得・評価を実行するアダプター。

    evaluate() は sklearn.base.clone() でモデルを複製するため、fitted 状態がリセットされ、
    元のモデルオブジェクトは変更されない。

    デフォルトパラメータ空間は提供しない。HPOAgent に param_space を必ず指定すること。

    Args:
        model: チューニング対象の sklearn 互換モデル（BaseEstimator のサブクラス）。
        eval_fn: 評価関数。シグネチャ: (model, X, y) -> float。大きいほど良いスコア。
        X: 特徴量データ。
        y: ターゲットデータ。
    """

    def __init__(
        self,
        model: BaseEstimator,
        eval_fn: Callable[..., float],
        X: Any,
        y: Any,
    ) -> None:
        """SklearnAdapter を初期化する。"""
        self._model = model
        self._eval_fn = eval_fn
        self._X = X
        self._y = y

    def get_default_param_space(self) -> ParamSpace:
        """sklearn はモデルが多様なためデフォルトパラメータ空間を提供しない。"""
        raise NotImplementedError(
            "SklearnAdapter はデフォルトパラメータ空間を持ちません。"
            " HPOAgent に param_space を指定してください。"
        )

    def evaluate(self, params: dict[str, Any]) -> float:
        """パラメータを設定してモデルを学習・評価し、スコアを返す。

        元のモデルオブジェクトは変更しない（sklearn.base.clone() を使用）。
        """
        model_copy: BaseEstimator = sklearn_clone(self._model)
        model_copy.set_params(**params)
        model_copy.fit(self._X, self._y)
        return self._eval_fn(model_copy, self._X, self._y)


class PyTorchAdapter(ModelAdapterBase):
    """PyTorch モデルに対してパラメータ空間取得・評価を実行するアダプター。

    PyTorch は fit/predict を持たないため、学習・評価ループ全体を eval_fn に委譲する。
    model_fn がパラメータを受け取ってモデルを構築し、eval_fn がそのモデルを学習・評価して
    スコアを返す。torch への依存はなく、任意の呼び出し可能オブジェクトを受け付ける。

    param_space は HPOAgent に渡すか、省略して LLM に自動生成させる。

    Args:
        model_fn: パラメータ辞書を受け取り、モデルオブジェクトを返すファクトリ関数。
        eval_fn: モデルオブジェクトを受け取り、学習・評価を行ってスコアを返す関数。
            大きいほど良いスコア。

    Example:
        >>> def model_fn(params):
        ...     return MyNet(hidden_size=params['hidden_size'])
        >>> def eval_fn(model):
        ...     # 学習・評価ループ
        ...     return accuracy
        >>> adapter = PyTorchAdapter(model_fn=model_fn, eval_fn=eval_fn)
    """

    def __init__(
        self,
        model_fn: Callable[[dict[str, Any]], Any],
        eval_fn: Callable[[Any], float],
    ) -> None:
        """PyTorchAdapter を初期化する。"""
        self._model_fn = model_fn
        self._eval_fn = eval_fn

    def get_default_param_space(self) -> ParamSpace:
        """PyTorch はデフォルトパラメータ空間を提供しない。LLM が自動生成する。"""
        raise NotImplementedError(
            "PyTorchAdapter はデフォルトパラメータ空間を持ちません。"
            " param_space を省略した場合は LLM が自動生成します。"
        )

    def evaluate(self, params: dict[str, Any]) -> float:
        """model_fn でモデルを構築し、eval_fn で学習・評価してスコアを返す。"""
        model = self._model_fn(params)
        return self._eval_fn(model)
