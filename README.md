# hpo-agent

AIエージェントがハイパーパラメーターチューニング（HPO）を自動化する Python ライブラリです。

LLM が最適化戦略を判断しながら、ベイズ最適化・準ランダム探索・専門家 AI による探索を組み合わせて、モデルの最良パラメーターを探索します。

---

## 目次

- [インストール](#インストール)
- [事前準備：LLM の設定](#事前準備llm-の設定)
- [基本的な使い方](#基本的な使い方)
- [入力パラメーター一覧](#入力パラメーター一覧)
- [出力（HPOResult）](#出力hporesult)
- [パラメーター空間のカスタマイズ](#パラメーター空間のカスタマイズ)
- [プロンプトのカスタマイズ](#プロンプトのカスタマイズ)
- [実験の再現性（乱数シード）](#実験の再現性乱数シード)
- [ログ・進捗の確認](#ログ進捗の確認)
- [動作環境](#動作環境)

---

## インストール

```bash
pip install git+https://github.com/your-org/hpo-agent.git
```

---

## 事前準備：LLM の設定

本ライブラリは LLM として Google Gemini を使用します。
プロジェクトのルートに `.env` ファイルを作成し、以下の環境変数を設定してください。

```ini
LLM_API_KEY=AIza...          # Google AI Studio で取得した API キー
LLM_MODEL_NAME=gemini-1.5-pro
LLM_PROVIDER=google
```

| 環境変数 | 説明 | 例 |
|---|---|---|
| `LLM_API_KEY` | 使用する LLM の API キー | `AIza...` |
| `LLM_MODEL_NAME` | 使用するモデル名 | `gemini-1.5-pro` |
| `LLM_PROVIDER` | LLM プロバイダー識別子 | `google` |

---

## 基本的な使い方

### LightGBM

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from hpo_agent import HPOAgent

# 1. データを用意
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. チューニング対象のモデルを用意
model = lgb.LGBMClassifier(verbosity=-1)

# 3. 評価関数を定義（スコアを返す。大きいほど良い）
#    eval_fn には学習済みモデルと X, y が渡される
def my_eval(model, X, y) -> float:
    return float(accuracy_score(y, model.predict(X)))

# 4. HPOAgent を作成して実行
agent = HPOAgent(
    model=model,
    eval_fn=my_eval,
    n_trials=50,
    X=X_train,
    y=y_train,
)

result = agent.run()

# 5. 結果を確認
print(result.best_params)   # 最良パラメーター辞書
print(result.best_score)    # 最良スコア
print(result.trials_df)     # 全試行の履歴（pandas DataFrame）
print(result.report)        # Markdown 形式のレポート
```

> **注意**
> `eval_fn` には、指定したパラメーターで学習済みのモデルと、`HPOAgent` に渡した `X`・`y` が渡されます。
> 「大きいほど良い」スコアを返す設計のため、RMSE や Log Loss など損失関数を使う場合は符号を反転してください。
> ```python
> def my_eval(model, X, y) -> float:
>     from sklearn.metrics import mean_squared_error
>     rmse = mean_squared_error(y, model.predict(X), squared=False)
>     return -rmse  # 符号反転して「大きいほど良い」に変換
> ```
>
> クロスバリデーションを使いたい場合は、`eval_fn` 内で `copy.deepcopy(model)` して再学習することも可能です（[s5e4_lgbm.py](example/s5e4_lgbm.py) を参照）。

### PyTorch

PyTorch モデルは `fit`/`predict` を持たないため、`model_fn`（モデルファクトリ）と `eval_fn`（学習・評価ループ全体）を渡す設計です。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from hpo_agent import HPOAgent, ParamSpace, ParamSpec

# 1. モデルファクトリを定義（パラメータ辞書を受け取りモデルを返す）
def model_fn(params):
    return nn.Sequential(
        nn.Linear(10, int(params["hidden_size"])),
        nn.ReLU(),
        nn.Linear(int(params["hidden_size"]), 1),
    )

# 2. 評価関数を定義（学習・評価ループ全体を含む。大きいほど良いスコア）
def eval_fn(model) -> float:
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    # ... 学習ループ ...
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()
    return -val_loss  # 損失を符号反転してスコアに変換

# 3. パラメーター空間を定義（PyTorch では必須）
param_space = ParamSpace(specs=(
    ParamSpec(name="hidden_size", type="int", low=16, high=256),
))

# 4. HPOAgent を作成して実行（X, y は不要）
agent = HPOAgent(
    model=model_fn,        # ファクトリ関数を渡す
    eval_fn=eval_fn,
    n_trials=20,
    param_space=param_space,
)

result = agent.run()
print(result.best_params)
```

> **PyTorch の注意点**
> - `model` 引数にはモデルインスタンスではなく **ファクトリ関数** を渡します
> - `eval_fn` のシグネチャは `(model) -> float`（`X`, `y` なし）
> - `param_space` の指定が必須です（デフォルト空間がありません）
> - `X`, `y` は `eval_fn` 内でキャプチャする設計のため、`HPOAgent` への引数は不要です

---

## 入力パラメーター一覧

```python
agent = HPOAgent(
    model=model,           # 必須
    eval_fn=my_eval,       # 必須
    n_trials=50,           # 必須
    X=X_train,             # LightGBM では必須、PyTorch では不要
    y=y_train,             # LightGBM では必須、PyTorch では不要
    param_space=None,      # PyTorch では必須、LightGBM では任意
    seed=42,               # 任意
    prompts={},            # 任意
    llm_model=None,        # 任意
)
```

| パラメーター | 型 | 必須 | 説明 |
|---|---|---|---|
| `model` | `Any` | Yes | チューニング対象モデル。LightGBM: `lgb.LGBMModel` のインスタンス。PyTorch: `(params: dict) -> model` のファクトリ関数 |
| `eval_fn` | `Callable` | Yes | 評価関数（大きいほど良いスコアを返す）。LightGBM: `(model, X, y) -> float`。PyTorch: `(model) -> float`（学習・評価ループ全体） |
| `n_trials` | `int` | Yes | HPO の総試行回数 |
| `X` | `Any` | LightGBM のみ | 特徴量データ（LightGBM 使用時は必須。PyTorch では不要） |
| `y` | `Any` | LightGBM のみ | ターゲットデータ（LightGBM 使用時は必須。PyTorch では不要） |
| `param_space` | `ParamSpace` | PyTorch では必須 | 探索するパラメーター空間。PyTorch では必須。LightGBM では省略時にデフォルト空間を使用 |
| `seed` | `int \| None` | No | 乱数シード。指定すると Sobol 探索・ベイズ最適化の結果が再現可能になる（デフォルト: `None`） |
| `prompts` | `dict[str, str]` | No | エージェント別の追加プロンプト（詳細は[プロンプトのカスタマイズ](#プロンプトのカスタマイズ)を参照） |
| `llm_model` | `str` | No | `.env` の `LLM_MODEL_NAME` を上書きしたい場合に指定 |

---

## 出力（HPOResult）

`agent.run()` は `HPOResult` オブジェクトを返します。

```python
result = agent.run()
```

| フィールド | 型 | 説明 |
|---|---|---|
| `result.best_params` | `dict` | 最良パラメーターの辞書 |
| `result.best_score` | `float` | 最良スコア |
| `result.trials_df` | `pd.DataFrame` | 全試行の履歴テーブル（下表参照） |
| `result.report` | `str` | Markdown 形式の最終レポート |

### trials_df のカラム

| カラム名 | 型 | 説明 |
|---|---|---|
| `trial_id` | `int` | 試行番号 |
| `params` | `dict` | 試行したパラメーター |
| `score` | `float` | `eval_fn` が返したスコア |
| `tool_used` | `str` | 使用した探索ツール名 |
| `timestamp` | `datetime` | 実行日時 |
| `eval_duration` | `float` | モデルの学習・評価にかかった時間（秒） |
| `algo_duration` | `float` | アルゴリズムが次の実験点を算出するのにかかった時間（秒） |
| `reasoning` | `str` | AI の判断理由（ツール選択理由またはパラメーター提案理由） |

### レポートの保存

```python
with open("hpo_report.md", "w") as f:
    f.write(result.report)
```

---

## パラメーター空間のカスタマイズ

デフォルトでは、LightGBM 用に以下の8パラメーターが探索されます。

| パラメーター | 探索範囲 | スケール |
|---|---|---|
| `num_leaves` | 20 〜 300 | 線形 |
| `max_depth` | 3 〜 12 | 線形 |
| `learning_rate` | 0.0001 〜 0.3 | 対数 |
| `n_estimators` | 50 〜 1000 | 線形 |
| `subsample` | 0.5 〜 1.0 | 線形 |
| `colsample_bytree` | 0.5 〜 1.0 | 線形 |
| `reg_alpha` | 1e-8 〜 10.0 | 対数 |
| `reg_lambda` | 1e-8 〜 10.0 | 対数 |

探索範囲を変更したい場合は、`ParamSpec` と `ParamSpace` を使って手動指定できます。

```python
from hpo_agent import HPOAgent, ParamSpec, ParamSpace

custom_space = ParamSpace(specs=(
    ParamSpec(name="num_leaves", type="int", low=50, high=200),
    ParamSpec(name="learning_rate", type="float", low=1e-3, high=0.1, log=True),
    ParamSpec(name="n_estimators", type="int", low=100, high=500),
))

agent = HPOAgent(
    model=model,
    eval_fn=my_eval,
    n_trials=50,
    param_space=custom_space,
)
```

### ParamSpec のフィールド

| フィールド | 説明 |
|---|---|
| `name` | パラメーター名（モデルの引数名と一致させる） |
| `type` | `"int"` / `"float"` / `"categorical"` |
| `low` | 数値型の下限値（`"int"` / `"float"` で必須） |
| `high` | 数値型の上限値（`"int"` / `"float"` で必須） |
| `choices` | 選択肢のタプル（`"categorical"` で必須） |
| `log` | `True` のとき対数スケールで探索（`low > 0` が必要） |

---

## プロンプトのカスタマイズ

各 AI エージェントにはデフォルトのシステムプロンプトが設定されています。
ドメイン知識や制約を追加したい場合は、`prompts` 引数で追記できます。

```python
agent = HPOAgent(
    model=model,
    eval_fn=my_eval,
    n_trials=50,
    prompts={
        "supervisor": "二値分類の精度改善を優先してください。",
        "expert_agent": "過学習しやすいデータのため正則化を重視してください。",
    }
)
```

| キー | 対象エージェント | 説明 |
|---|---|---|
| `"supervisor"` | スーパーバイザー | 全体の探索戦略に関する指示を追加する |
| `"expert_agent"` | ExpertAgentTool | 専門家 AI によるパラメーター提案に関する指示を追加する |

ユーザープロンプトはデフォルトのシステムプロンプトの末尾に連結されます。

---

## 実験の再現性（乱数シード）

`seed` を指定すると、同じ設定での実行結果が再現可能になります。

```python
agent = HPOAgent(
    model=model,
    eval_fn=my_eval,
    n_trials=50,
    seed=42,
)
```

| コンポーネント | シードの効果 |
|---|---|
| `SobolSearchTool` | Sobol 列の生成が固定される |
| `BayesianOptimizationTool` | Optuna TPE サンプラーの乱数が固定される |
| `ExpertAgentTool` | LLM の出力は確率的なため完全な再現は保証されない |

> 使用したシード値はレポートのヘッダーに記載されます。`seed=None`（デフォルト）の場合は非決定的な動作になります。

---

## ログ・進捗の確認

実行中はコンソールにログが出力されます。

```
[INFO]  sobol_search 完了 | 試行数: 10 | 最良スコア: 0.8712 | 実行時間: 12.3s
--- 中間レポート ---
現時点の最良スコア: 0.8712
...
```

ログレベルは Python 標準の `logging` モジュールで制御できます。

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # 各試行の詳細ログを表示
```

| ログレベル | 出力内容 |
|---|---|
| `INFO`（デフォルト） | ツール完了時のサマリーと中間レポート |
| `DEBUG` | 各試行の番号・スコア・パラメーター・評価時間・アルゴリズム計算時間 |

---

## 動作環境

| 項目 | 要件 |
|---|---|
| Python | 3.12 以上 |
| OS | Linux / macOS / Windows |

### 主な依存パッケージ

| パッケージ | 用途 |
|---|---|
| `langchain` / `langgraph` | エージェント・ツール実装 |
| `optuna` | ベイズ最適化 |
| `lightgbm` | MVP 対象モデル |
| `pandas` | 試行履歴テーブル |
| `python-dotenv` | `.env` 読み込み |
| `pydantic` | スキーマ定義 |
