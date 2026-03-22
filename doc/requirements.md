# hpo-agent — Pythonパッケージ 要件定義書

Version 0.1 | 2026年3月

---

## 1. 概要

### 1.1 背景・目的

ハイパーパラメータチューニング（HPO）は機械学習モデルの性能向上に不可欠だが、初心者にとって以下の課題がある。

- 探索すべきパラメータが不明
- パラメータの探索範囲が不明
- 探索手法の選択が困難

hpo-agent は、AIエージェントが HPO を自動化することでこれらの課題を解消し、初心者でも高品質なモデルチューニングを実現できる Python ライブラリである。

### 1.2 提供形態

- Python ライブラリとして提供（アプリケーション開発は対象外）
- GitHub 経由で配布（`pip install git+...` によるインストール）
- パッケージ名：`hpo-agent`

---

## 2. スコープ

### 2.1 初期対応モデル

MVP（最小実装）として LightGBM を対象とする。

| フェーズ | 対応モデル | 備考 |
|---|---|---|
| MVP | LightGBM | 初期評価・動作検証 |
| 対応済み | scikit-learn 系モデル | fit/predict インターフェース準拠。param_space の指定が必須 |
| 対応済み | PyTorch | `model_fn: (params) -> model` + `eval_fn: (model) -> float` で対応。param_space の指定が必須。 |

### 2.2 対象外

- GUI / Web アプリケーション
- 分散実行・並列化（初期バージョン）
- モデルのデプロイ・サービング

---

## 3. ユーザーインターフェース（API 仕様）

### 3.1 基本的な使い方

```python
from hpo_agent import HPOAgent

def my_eval(model, X, y) -> float:
    # スコアを返すカスタム評価関数
    ...

agent = HPOAgent(
    model=lgbm_model,
    eval_fn=my_eval,
    n_trials=50,
    prompts={
        "supervisor": "二値分類の精度改善を優先してください。",
        "expert_agent": "過学習しやすいデータのため正則化を重視。",
    }
)

result = agent.run()          # 同期実行
print(result.report)          # Markdown レポート
print(result.trials_df)       # pandas DataFrame
print(result.best_params)     # 最良パラメータ
```

### 3.2 入力パラメータ

| パラメータ | 型 | 必須 | 説明 |
|---|---|---|---|
| `model` | `Any` | Yes | チューニング対象のモデルオブジェクト |
| `X` | `Any` | No | 学習に使用する特徴量データ。LightGBM / sklearn モデルでは必須。PyTorch モデルでは不要。 |
| `y` | `Any` | No | 学習に使用するターゲットデータ。LightGBM / sklearn モデルでは必須。PyTorch モデルでは不要。 |
| `eval_fn` | `Callable` | Yes | ユーザー定義の評価関数。スコア（float）を返す |
| `n_trials` | `int` | Yes | HPO の総試行回数 |
| `param_space` | `ParamSpace` | No | 最適化対象のハイパーパラメーターを指定する。指定がない場合はモデルアダプターのデフォルト空間を使用する |
| `seed` | `int \| None` | No | 乱数シード。指定すると Sobol 列・ベイズ最適化サンプラーの乱数が固定され、実験が再現可能になる。`None`（デフォルト）の場合は非決定的 |
| `prompts` | `dict[str, str]` | No | エージェント別のユーザープロンプト（後述） |
| `llm_model` | `str` | No | `.env` の LLM モデル名を上書きしたい場合に指定 |

### 3.3 出力（HPOResult）

| フィールド | 型 | 説明 |
|---|---|---|
| `best_params` | `dict` | 最良パラメータの辞書 |
| `best_score` | `float` | 最良スコア |
| `trials_df` | `pd.DataFrame` | 全試行の履歴テーブル |
| `report` | `str` | Markdown 形式のテキストレポート |

---

## 4. アーキテクチャ

### 4.1 全体構成

```
ユーザー
  └─ HPOAgent
       └─ スーパーバイザー（LangGraph）
            ├─ BayesianOptimizationTool
            ├─ SobolSearchTool
            └─ ExpertAgentTool（専門家 AI エージェント）
                 └─ HPOResult（DataFrame + Markdown レポート）
```

### 4.2 スーパーバイザーエージェント

- LangChain / LangGraph で実装
- LLM が各ツールの実行順序・試行回数を動的に決定
- 各ツールの結果をもとに次の打ち手を判断するループを構成

### 4.3 ツール一覧

| ツール名 | 概要 | 優先度 |
|---|---|---|
| `BayesianOptimizationTool` | ベイズ最適化による探索（Optuna 等を利用） | MVP |
| `SobolSearchTool` | Sobol 列による準ランダム探索 | MVP |
| `ExpertAgentTool` | 専門家 AI エージェントによる決め打ち探索 | MVP |
| `NarrowSearchSpaceTool` | 過去の探索結果をもとに探索空間を動的に狭める | MVP |

> ツールはインターフェースを定義し、追加・差し替えが容易な設計とする。

---

## 5. LLM 設定

### 5.1 初期対応 LLM

MVP では Google Gemini を使用する。

### 5.2 設定方法

`.env` ファイルにユーザーが API キーとモデル名を設定することを前提とする。

| 環境変数 | 説明 | 例 |
|---|---|---|
| `LLM_API_KEY` | 使用する LLM の API キー | `AIza...` |
| `LLM_MODEL_NAME` | 使用するモデル名 | `gemini-1.5-pro` |
| `LLM_PROVIDER` | LLM プロバイダー識別子 | `google` |

### 5.3 拡張性

- LLM プロバイダーを抽象化したインターフェースを定義
- OpenAI、Anthropic、ローカル LLM 等への切り替えを将来サポート

---

## 6. 出力仕様

### 6.1 試行履歴テーブル（trials_df）

pandas DataFrame として返却。以下のカラムを含む（最低限）。

| カラム名 | 型 | 説明 |
|---|---|---|
| `trial_id` | `int` | 試行番号 |
| （各パラメータ名） | `int` / `float` / `str` | 各パラメータが個別カラムとして展開される（例: `num_leaves`, `learning_rate`） |
| `score` | `float` | eval_fn が返したスコア |
| `tool_used` | `str` | 使用したツール名 |
| `timestamp` | `str` | 実行日時（ISO 8601 形式） |
| `eval_duration` | `float` | モデルの学習・評価にかかった時間（秒） |
| `algo_duration` | `float` | アルゴリズムが次の実験点を算出するのにかかった時間（秒）。ベイズ最適化なら TPE の提案計算時間、Sobol なら列生成時間、ExpertAgentTool なら LLM API 呼び出し時間 |
| `reasoning` | `str` | ExpertAgentTool のパラメーター提案理由（他ツールは空文字） |

### 6.2 テキストレポート（report）

レポートは **中間レポート** と **最終レポート** の2種類を提供する。

#### 中間レポート

- ツールの実行が完了するたびにコンソールへ出力する（`logging.INFO`）
- LLM を使用せず、現時点の統計情報のみを含む
- 含む内容：
  - 現時点の最良パラメータ・スコア
  - スコア推移
  - 使用ツールと試行回数の内訳
  - スーパーバイザーが今回のツールを選択した理由（AI の判断根拠）
  - ExpertAgentTool を使用した場合はパラメーター提案の理由
  - 今回のツール実行で消費した合計時間（モデル評価時間・アルゴリズム計算時間の内訳）

#### 最終レポート（`HPOResult.report`）

- 全試行完了後に `HPOResult.report` として返却する Markdown 文字列
- 含む内容：
  - 最良パラメータのサマリー
  - スコア推移の考察
  - 使用ツールと試行回数の内訳
  - 各ステップでの AI の判断理由の記録（ツール選択理由・パラメーター提案理由）
  - AI エージェントによる最終考察・推薦コメント（LLM 生成）
  - 全試行の時間サマリー（総最適化時間・ツール別平均モデル評価時間・ツール別平均アルゴリズム計算時間）

---

## 7. ログ・進捗表示

- Python 標準 `logging` モジュールを使用
- 実行中の進捗をコンソールに出力（試行番号・スコア・使用ツール等）
- **各試行のログ**（`logging.DEBUG`）：試行番号・スコア・パラメータ・モデル評価時間・アルゴリズム計算時間
- **ツール完了時のログ**（`logging.INFO`）：ツール名・実行試行数・今回の最良スコア・ツール合計実行時間、続けて中間レポートを出力
- ログレベルは `INFO` / `DEBUG` で切り替え可能
- ユーザーが logging 設定を上書きできる設計

---

## 8. 非機能要件

### 8.1 拡張性

- モデルは sklearn 互換インターフェース（fit/predict）を基本とし、カスタム eval 関数で任意のモデルに対応
- ツールはプラグイン的に追加可能な設計
- LLM プロバイダーは抽象化レイヤーで切り替え可能

### 8.2 実行方式

- 同期実行（`run()` が完了するまでブロック）
- 将来的な非同期対応を見据えた設計

### 8.3 再現性（乱数シード管理）

- `HPOAgent` の `seed` 引数でユーザーが乱数シードを指定できる
- シードが指定された場合、以下のコンポーネントに伝播させることで**同一シード・同一設定での実行結果が再現可能**になる

| コンポーネント | シードの使用 |
|---|---|
| `SobolSearchTool` | `scipy.stats.qmc.Sobol(seed=seed)` に渡す |
| `BayesianOptimizationTool` | `optuna.samplers.TPESampler(seed=seed)` に渡す |
| `ExpertAgentTool` | LLM の出力は確率的で完全な再現は保証しない（`temperature` 設定に依存） |

- シードが `None`（デフォルト）の場合は非決定的な挙動になる
- 使用したシード値はレポートのヘッダーに記載し、ユーザーが後から確認できるようにする

---

## 9. 開発・品質管理

| 項目 | ツール・方針 |
|---|---|
| ユニットテスト | pytest（カバレッジ計測含む） |
| 型チェック | mypy（strict モードを推奨） |
| コードフォーマット | ruff（lint）+ black（フォーマット） |
| CI | GitHub Actions（push / PR 時に自動実行） |
| バージョン管理 | GitHub（`pip install git+...` で配布） |

### 9.1 CI パイプライン（GitHub Actions）

- ruff / black によるフォーマットチェック
- mypy による型チェック
- pytest によるテスト実行
- Python 3.12 のマトリクステスト（将来的に 3.13 も追加予定）

---

## 10. プロンプトカスタマイズ

### 10.1 設計方針

各エージェントにはライブラリが定義するデフォルトのシステムプロンプトを持つ。ユーザーはオプションで追加プロンプト（ユーザープロンプト）を渡すことができ、デフォルトプロンプトに追記される形で動作する。

### 10.2 プロンプト設定対象

| 対象 | 説明 | 設定キー |
|---|---|---|
| スーパーバイザー | 全体の最適化戦略を司るエージェント | `supervisor` |
| ExpertAgentTool | 専門家 AI による決め打ち探索ツール内エージェント | `expert_agent` |
| （拡張時）他ツール内エージェント | 将来追加されるエージェント型ツール | 任意キーで拡張 |

### 10.3 入力インターフェース

`HPOAgent` に `prompts` を辞書形式で渡す。

```python
agent = HPOAgent(
    model=lgbm_model,
    eval_fn=my_eval,
    n_trials=50,
    prompts={
        "supervisor": "二値分類の精度改善を優先してください。",
        "expert_agent": "過学習しやすいデータのため正則化を重視。",
    }
)
```

### 10.4 プロンプトの結合ルール

| レイヤー | 内容 | 編集可否 |
|---|---|---|
| デフォルトシステムプロンプト | ライブラリが定義する基本動作・制約 | 不可（ライブラリ管理） |
| ユーザープロンプト | ユーザーが追記するドメイン知識・制約 | 自由記述 |
| 結合方式 | デフォルト + ユーザープロンプトを連結して LLM に渡す | — |

---

## 11. 主要依存パッケージ（予定）

| パッケージ | 用途 |
|---|---|
| `langchain` / `langgraph` | エージェント・ツール実装 |
| `optuna` | ベイズ最適化 |
| `lightgbm` | MVP 対象モデル |
| `pandas` | 試行履歴テーブル |
| `python-dotenv` | `.env` 読み込み |
| `pydantic` | 入出力スキーマ定義 |

---

## 12. 動作環境

| 項目 | 要件 |
|---|---|
| Python バージョン | 3.12 以上 |
| OS | Linux / macOS / Windows（CI は Linux ベース） |
| パッケージ管理 | pip（将来的に uv / poetry 対応も検討） |

---

## 13. 今後の展望

- scikit-learn 系モデルへの対応拡張
- PyTorch / TensorFlow 対応
- 非同期実行（async/await）対応
- 並列探索（複数ツールの同時実行）
- Optuna Dashboard 等との可視化連携
- PyPI への公開