# HPO-Agent

## Project Overview

AIを用いたハイパーパラメーターチューニングエージェントの開発プロジェクト。

## Tech Stack

- **Language**: Python (PEP8準拠)
- **Package Manager**: uv
- **Version Control**: Git / GitHub
- **GitHub CLI**: `gh` コマンドを使用

## Python Coding Standards

### Docstrings

- すべての関数・クラスにDocstringを記載する
- **クラス**: 詳細な説明（目的・責務・使用例など）
- **メソッド・関数**: 簡潔な説明（何をするかを1〜2行で）

### Type Hints

- すべての関数・メソッドの引数と戻り値に型ヒントを必ず付ける
- mypy strict モードで型チェックを行う

### Style

- PEP8に準拠する
- formatter: Black
- linter: ruff

## Package Management (uv)

```bash
# 依存関係の追加
uv add <package>

# 開発依存関係の追加
uv add --dev <package>

# 依存関係のインストール
uv sync

# スクリプトの実行
uv run python <script.py>
```

## Quality Management

| 項目 | ツール・方針 |
|---|---|
| ユニットテスト | pytest（カバレッジ計測含む） |
| 型チェック | mypy（strict モード） |
| linter | ruff |
| formatter | Black |
| CI | GitHub Actions（push / PR 時に自動実行） |

### CI パイプライン（GitHub Actions）

- ruff / Black によるフォーマットチェック
- mypy による型チェック
- pytest によるテスト実行
- Python 3.12 でのマトリクステスト

## LLM Configuration

`.env` ファイルにAPIキーとモデル名を設定する。

| 環境変数 | 説明 | 例 |
|---|---|---|
| `LLM_API_KEY` | 使用する LLM の API キー | `AIza...` |
| `LLM_MODEL_NAME` | 使用するモデル名 | `gemini-1.5-pro` |
| `LLM_PROVIDER` | LLM プロバイダー識別子 | `google` |

- MVP では Google Gemini を使用
- LLM プロバイダーは抽象化レイヤーで切り替え可能（OpenAI / Anthropic / ローカル LLM）
- `HPOAgent` の `llm_model` 引数で `.env` の設定を上書き可能

## Architecture

```
ユーザー
  └─ HPOAgent
       └─ スーパーバイザー（LangGraph）
            ├─ BayesianOptimizationTool  # Optuna によるベイズ最適化
            ├─ SobolSearchTool           # Sobol 列による準ランダム探索
            └─ ExpertAgentTool           # 専門家 AI エージェントによる決め打ち探索
                 └─ HPOResult（DataFrame + Markdown レポート）
```

- ツールはインターフェースを定義し、追加・差し替えが容易な設計とする
- モデルは sklearn 互換インターフェース（fit/predict）を基本とし、カスタム `eval_fn` で任意のモデルに対応

## Key Dependencies

| パッケージ | 用途 |
|---|---|
| `langchain` / `langgraph` | エージェント・ツール実装 |
| `optuna` | ベイズ最適化 |
| `lightgbm` | MVP 対象モデル |
| `pandas` | 試行履歴テーブル |
| `python-dotenv` | `.env` 読み込み |
| `pydantic` | 入出力スキーマ定義 |

## Runtime Requirements

- Python: 3.12 以上
- OS: Linux / macOS / Windows（CI は Linux ベース）

## Git Rules
詳細は @.claude/rules/git-rules.md を参照

## GitHub Workflow

- GitHub との通信は `gh` コマンドを使用する
- PRの作成: `gh pr create`
- Issueの確認: `gh issue list`
- CIの確認: `gh run list`

## Documentation Consistency

開発途中に変更が生じた場合は、必ず関連するドキュメントを確認し、コードとドキュメントが一致していることを確認する。

## Directory Structure
```
HPO-Agent/
├── src/hpo_agent/       # パッケージソースコード
├── example/             # 使用例コード
├── test/                # テストコード
├── doc/
│   ├── requirements.md          # 要件定義書
│   ├── impl_design.md           # 実装設計書
│   ├── model_adapter_design.md  # モデルアダプター設計書
│   ├── llm_prompt_design.md     # LLM プロンプト設計書
│   └── test_design.md           # テスト設計書
├── .github/workflows/   # GitHub Actions CI 設定
├── .env.example         # 環境変数テンプレート
├── pyproject.toml       # プロジェクト設定・依存関係
└── README.md            # ユーザー向けドキュメント
```