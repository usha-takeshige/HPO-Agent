# hpo-agent

AIエージェントがハイパーパラメーターチューニング（HPO）を自動化する Python ライブラリです。

LLM が最適化戦略を判断しながら、ベイズ最適化・準ランダム探索・専門家 AI による探索を組み合わせ、探索空間の動的な絞り込みも行いながら、モデルの最良パラメーターを探索します。

---

## 目次

- [インストール](#インストール)
- [事前準備：LLM の設定](#事前準備llm-の設定)
- [基本的な使い方](#基本的な使い方)
- [最適化ツール](#最適化ツール)
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

# 2. 評価関数を定義（パラメーター辞書を受け取りスコアを返す。大きいほど良い）
#    X_train / y_train はクロージャでキャプチャする
def eval_fn(params: dict) -> float:
    model = lgb.LGBMClassifier(verbosity=-1, **params)
    model.fit(X_train, y_train)
    return float(accuracy_score(y_val, model.predict(X_val)))

# 3. HPOAgent を作成して実行
agent = HPOAgent(
    eval_fn=eval_fn,
    n_trials=50,
)

result = agent.run()

# 4. 結果を確認
print(result.best_params)   # 最良パラメーター辞書
print(result.best_score)    # 最良スコア
print(result.trials_df)     # 全試行の履歴（pandas DataFrame）
print(result.report)        # Markdown 形式のレポート
```

> **注意**
> `eval_fn` はパラメーター辞書 `params: dict` を受け取り、スコア（`float`）を返す関数です。
> 「大きいほど良い」スコアを返す設計のため、RMSE や Log Loss など損失関数を使う場合は符号を反転してください。
> ```python
> def eval_fn(params: dict) -> float:
>     from sklearn.metrics import root_mean_squared_error
>     model = lgb.LGBMRegressor(**params).fit(X_train, y_train)
>     rmse = root_mean_squared_error(y_val, model.predict(X_val))
>     return -rmse  # 符号反転して「大きいほど良い」に変換
> ```
>
> クロスバリデーションを使いたい場合は、`eval_fn` 内でループを実装します（[s5e4_lgbm.py](example/s5e4_lgbm.py) を参照）。

### PyTorch

PyTorch モデルは `fit`/`predict` を持たないため、`eval_fn` の中でモデルの構築・学習・評価をすべて行います。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from hpo_agent import HPOAgent, ParamSpace, ParamSpec

# X_train, y_train, X_val, y_val は事前に用意しておく

# 1. 評価関数を定義（モデル構築・学習・評価ループ全体を含む）
def eval_fn(params: dict) -> float:
    model = nn.Sequential(
        nn.Linear(10, int(params["hidden_size"])),
        nn.ReLU(),
        nn.Linear(int(params["hidden_size"]), 1),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train).squeeze(1), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val).squeeze(1), y_val).item()
    return -val_loss  # 損失を符号反転してスコアに変換

# 2. HPOAgent を作成して実行（param_space 省略時は LLM が自動生成）
agent = HPOAgent(
    eval_fn=eval_fn,
    n_trials=20,
    # param_space を省略すると LLM が eval_fn のソースコードを読んで自動設計する
)

result = agent.run()
print(result.best_params)
```

> **PyTorch の注意点**
> - `eval_fn` のシグネチャは `(params: dict) -> float`。モデルの構築から評価まで `eval_fn` 内に実装します
> - `X_train` / `y_train` など学習データは `eval_fn` のクロージャでキャプチャします
> - `param_space` は省略可能です。省略した場合は LLM が `eval_fn` のソースコードをもとに探索空間を自動設計します

---

## 最適化ツール

スーパーバイザー（LLM）が以下の4種類のツールを自動的に選択・組み合わせながら探索を進めます。ユーザーが直接ツールを指定する必要はありません。

| ツール | 概要 | 主な用途 |
|---|---|---|
| `SobolSearchTool` | Sobol 列による準ランダム探索 | 探索序盤で空間を均一にカバーする |
| `BayesianOptimizationTool` | Optuna TPE サンプラーによるベイズ最適化 | これまでの試行結果を活用して効率的に探索する |
| `ExpertAgentTool` | 専門家 AI エージェントによる決め打ち探索 | 試行履歴を分析し、有望なパラメーターを直接提案する |
| `ChangeSearchSpaceTool` | 探索空間の動的変更 | 有望な範囲に絞り込んだり、狭めすぎた空間を広げたりする |

> **補足**
> `ChangeSearchSpaceTool` は LLM が探索空間の調整が必要と判断したタイミングで自動呼び出されます。
> 以降の探索（`SobolSearchTool` / `BayesianOptimizationTool` / `ExpertAgentTool`）は、変更後の空間を対象に動作します。

---

## 入力パラメーター一覧

```python
agent = HPOAgent(
    eval_fn=eval_fn,       # 必須
    n_trials=50,           # 必須
    param_space=None,      # 任意（省略時は LLM が自動生成）
    seed=42,               # 任意
    prompts={},            # 任意
    llm_model=None,        # 任意
)
```

| パラメーター | 型 | 必須 | 説明 |
|---|---|---|---|
| `eval_fn` | `Callable[[dict], float]` | Yes | 評価関数。`params: dict` を受け取り、スコア（大きいほど良い）を返す。モデルの学習・評価・データの受け渡しはすべてこの関数内で行う |
| `n_trials` | `int` | Yes | HPO の総試行回数 |
| `param_space` | `ParamSpace` | No | 探索するパラメーター空間。省略時は LLM が `eval_fn` のソースコード・試行回数をもとに自動設計する |
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
| `result.report` | `str` | Markdown 形式の最終レポート（下記参照） |

### trials_df のカラム

| カラム名 | 型 | 説明 |
|---|---|---|
| `trial_id` | `int` | 試行番号 |
| `<param_name>` | `int` / `float` / `str` | 試行したパラメーター（各パラメーターが個別カラムとして展開される。例: `num_leaves`, `learning_rate`） |
| `score` | `float` | `eval_fn` が返したスコア |
| `tool_used` | `str` | 使用した探索ツール名 |
| `timestamp` | `str` | 実行日時（ISO 8601 形式） |
| `eval_duration` | `float` | モデルの学習・評価にかかった時間（秒） |
| `algo_duration` | `float` | アルゴリズムが次の実験点を算出するのにかかった時間（秒） |
| `reasoning` | `str` | AI の判断理由（ツール選択理由またはパラメーター提案理由） |

### レポートの内容

`result.report` は Markdown 形式で以下のセクションを含みます。

- **最良結果サマリー**：最良パラメーターとスコア
- **スコア推移**：試行ごとのスコア変化
- **ツール使用内訳**：各ツールの使用回数
- **探索空間の変更履歴**（`ChangeSearchSpaceTool` が呼ばれた場合）：変更回数・各変更時点の試行数・変更前後の探索空間
- **AI 判断理由の記録**：各ステップでのツール選択・パラメーター提案の根拠
- **全試行の時間サマリー**：総最適化時間・ツール別平均実行時間
- **AI 考察**：LLM による最適化傾向の分析・改善提案

### レポートの保存

```python
with open("hpo_report.md", "w") as f:
    f.write(result.report)
```

---

## パラメーター空間のカスタマイズ

`param_space` を省略した場合、LLM が `eval_fn` のソースコード・試行回数をもとに、適切な探索空間を自動設計します。LightGBM・sklearn・PyTorch を問わず、任意のモデルで利用できます。

探索空間を明示的に指定したい場合は、`ParamSpec` と `ParamSpace` を使って手動指定できます。

```python
from hpo_agent import HPOAgent, ParamSpec, ParamSpace

custom_space = ParamSpace(specs=(
    ParamSpec(name="num_leaves", type="int", low=50, high=200),
    ParamSpec(name="learning_rate", type="float", low=1e-3, high=0.1, log=True),
    ParamSpec(name="n_estimators", type="int", low=100, high=500),
))

agent = HPOAgent(
    eval_fn=eval_fn,
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

### モデルの説明を渡す（推奨）

LLM は `eval_fn` のソースコードからパラメーター名のみを手がかりにモデルを推論します。
**`prompts` でモデルの種類・タスク・データの特性を説明することで、より適切な探索空間の設計や戦略の選択が期待できます。**

```python
agent = HPOAgent(
    eval_fn=eval_fn,
    n_trials=50,
    prompts={
        "supervisor": (
            "モデル: LGBMClassifier\n"
            "タスク: 二値分類（Titanic 生存予測）\n"
            "特徴量: 7 列（数値・カテゴリカル混在）、サンプル数: 712\n"
            "評価指標: 正解率（大きいほど良い）\n"
            "過学習が起きやすいため、正則化パラメーターを重視してください。"
        ),
    }
)
```

### その他のカスタマイズ例

```python
agent = HPOAgent(
    eval_fn=eval_fn,
    n_trials=50,
    prompts={
        "supervisor": "探索後半はベイズ最適化を優先してください。",
        "expert_agent": "learning_rate は 0.05 以下を優先的に試してください。",
    }
)
```

| キー | 対象エージェント | 説明 |
|---|---|---|
| `"supervisor"` | スーパーバイザー | 全体の探索戦略に関する指示を追加する。モデルやタスクの説明もここに書く |
| `"expert_agent"` | ExpertAgentTool | 専門家 AI によるパラメーター提案に関する指示を追加する |

ユーザープロンプトはデフォルトのシステムプロンプトの末尾に連結されます。

---

## 実験の再現性（乱数シード）

`seed` を指定すると、同じ設定での実行結果が再現可能になります。

```python
agent = HPOAgent(
    eval_fn=eval_fn,
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
