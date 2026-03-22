---
name: test-design
description: |
  hpo-agentの実装設計書をもとに、再現実装の検証に必要なテスト設計を行う。
  テストコードの骨格（pytest）と、グラフによる振る舞い確認の観点を出力する。
  以下のような場面で必ず使用すること：
  - 「テスト設計をしてください」
  - 「テストケースを作ってください」
  - 「エージェントの動作を確認するテストを書いてください」
  - 「HPOの最適化性能を確認するテストを設計してください」
---

# Skill: hpo-agent テスト設計

## 概要

`/doc/requirements.md`（要件定義書）と `/doc/imp_design.md`（実装設計書）の出力をもとに、以下を構造化して出力する。

- **単体テスト（pytest）**：各コンポーネントの正確性テスト・HPOとして期待される性質のテスト
- **振る舞いテスト**：グラフ出力を通じて確認すべき視覚的な観点の一覧

テストの実行・コードの生成は**このSkillのスコープ外**。テスト設計の構造化とコードの骨格作成に集中する。

---

## トリガー

ユーザーが以下のような指示を出したとき、このSkillを使用する。

- 「テスト設計をしてください」
- 「テストケースを作ってください」
- 「エージェントの動作を確認するテストを書いてください」
- 「HPOの最適化性能を確認するテストを設計してください」

---

## 入力ファイル

以下のファイルを優先順位の高い順に参照する。

| 優先順位 | ファイル | 内容 |
|---------|---------|------|
| 1（必須） | `/doc/imp_design.md` | クラス設計・パブリックAPI・データクラス定義・各ツールの仕様 |
| 2（必須） | `/doc/requirements.md` | パッケージ要件定義・入出力仕様・アーキテクチャ概要 |

---

## テストの2分類

テストは以下の2種類に明確に分けて設計・実装する。ファイル・クラス・マーカーの命名で区別を徹底する。

### 分類A：単体テスト（pytest）

pytestで自動実行できる形式のテスト。さらに以下の3つのサブカテゴリに分ける。

| サブカテゴリ | 目的 | pytestマーカー | ファイル名 |
|------------|------|--------------|----------|
| **A-1. コンポーネント実装テスト** | 各クラス・メソッドが仕様通りに動作するか | `@pytest.mark.component` | `test_components.py` |
| **A-2. エージェント動作テスト** | スーパーバイザーとツール群が連携して正しく動作するか | `@pytest.mark.agent` | `test_agent.py` |
| **A-3. HPO性質テスト** | HPOとして期待される最適化の性質が成立するか | `@pytest.mark.hpo` | `test_hpo_properties.py` |

**各サブカテゴリの違い**

- **A-1** は「個々のクラス・メソッドが正しく実装されているか」を検証する。`HPOAgent` の初期化・`HPOResult` の構造・各ツールの入出力・プロンプトの結合ルールなど。LLM呼び出しはモックに置き換える。
- **A-2** は「スーパーバイザーとツール群が正しく連携するか」を検証する。LangGraph のグラフ構造・ツール選択ロジック・試行回数の制御・ログ出力など。LLM呼び出しはモックに置き換え、エージェントの制御フローを確認する。
- **A-3** は「HPOとして期待される最適化の性質が成立するか」を検証する。ランダムより良いスコアが得られるか・ベイズ最適化が反復で改善するか・試行回数が守られるかなど、最適化器として最低限保証すべき性質をテストに落とし込む。軽量なダミーモデルと確定的な eval 関数を使う。

### 分類B：振る舞いテスト（グラフ出力）

自動判定が困難な視覚的・定性的な性質を、グラフを出力して人間が確認するテスト。pytestと分離し、独立したスクリプトとして定義する。

| 項目       | 内容                                                               |
| ---------- | ------------------------------------------------------------------ |
| ファイル名 | `check_behavior.py`                                                |
| 実行方法   | `uv run python tests/check_behavior.py` で単独実行                 |
| 出力形式   | matplotlib などでグラフを出力し、確認観点をタイトル・注釈として表示 |

---

## 出力フォーマット

以下の構造でMarkdownを出力する。
出力先：`doc/test_design.md`

---

### 出力テンプレート

````markdown
# テスト設計書：hpo-agent

## 1. テスト設計の方針

**参照ファイル**：`/doc/imp_design.md`、`/doc/requirements.md`

{テスト設計で特に重視する観点を2〜3文で記述。LLMエージェントのモック戦略・HPO性能の検証方針・プロンプト結合の確認方針を明示する。}

---

## 2. モック戦略

> LLM呼び出しを含むテストでは、実際のAPIを叩かずにモックを使う。
> モックの種類と使い分けを定義する。

| モック名 | 対象 | 振る舞い | 使用するテスト |
|---------|------|---------|--------------|
| `MockLLM` | LLM呼び出し全般 | 固定のツール選択・試行回数を返す | A-1, A-2 |
| `MockSupervisor` | スーパーバイザー全体 | 固定の実行計画を返す | A-1（ツール単体テスト） |
| `DummyModel` | チューニング対象モデル | 軽量なsklearn互換モデル | A-1, A-2, A-3 |
| `DeterministicEvalFn` | eval 関数 | パラメータの関数として確定的なスコアを返す | A-3 |

---

## 3. 単体テスト：A-1 コンポーネント実装テスト（`test_components.py`）

> 各クラス・メソッドが仕様通りに動作するかを検証する。
> `@pytest.mark.component` マーカーを付与する。
> LLM呼び出しはすべてモックに置き換える。

### テストケース一覧

| テストID | 対象クラス / メソッド | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|--------------------|---------|---------|--------------------|
| CMP-01 | `HPOAgent.__init__` | 必須引数で正常に初期化される | `model`, `eval_fn`, `n_trials` を渡す | 例外が発生しない |
| CMP-02 | `HPOAgent.__init__` | `prompts` 省略時にデフォルト値が設定される | `prompts` を渡さない | `prompts` が空辞書またはデフォルト値になる |
| CMP-03 | `HPOAgent.run` | 戻り値が `HPOResult` 型である | MockLLM を使用して `run()` を呼ぶ | `isinstance(result, HPOResult)` が True |
| CMP-04 | `HPOResult` | 全フィールドが正しい型を持つ | `HPOResult` を生成する | `best_params: dict`, `best_score: float`, `trials_df: pd.DataFrame`, `report: str` |
| CMP-05 | `HPOResult.trials_df` | 必須カラムをすべて含む | n_trials=5 で実行 | `trial_id`, `params`, `score`, `tool_used`, `timestamp` カラムが存在する |
| CMP-06 | `HPOResult.report` | Markdown 形式の文字列が返る | `run()` を実行 | `#` または `##` を含む文字列が返る |
| CMP-07 | `BayesianOptimizationTool` | 指定した試行回数だけ eval_fn が呼ばれる | `n_trials=3` で実行 | eval_fn の呼び出し回数が 3 回 |
| CMP-08 | `SobolSearchTool` | 指定した試行回数だけ eval_fn が呼ばれる | `n_trials=4` で実行 | eval_fn の呼び出し回数が 4 回 |
| CMP-09 | `ExpertAgentTool` | LLM から受け取ったパラメータで eval_fn が呼ばれる | MockLLM が固定パラメータを返す | eval_fn に渡されたパラメータが MockLLM の出力と一致 |
| CMP-10 | プロンプト結合 | デフォルト + ユーザープロンプトが連結される | `prompts={"supervisor": "追加指示"}` | LLM に渡されるプロンプトがデフォルト文字列 + "追加指示" を含む |
| CMP-11 | プロンプト結合 | `prompts` に存在しないキーは無視される | `prompts={"unknown_key": "..."}` | 例外が発生せず、デフォルトプロンプトのみ使用される |
| CMP-12 | `HPOAgent` | `n_trials` が総試行回数の上限として機能する | `n_trials=10`, MockLLM を使用 | eval_fn の総呼び出し回数が 10 以下 |

---

## 4. 単体テスト：A-2 エージェント動作テスト（`test_agent.py`）

> スーパーバイザーとツール群の連携・LangGraph の制御フローを検証する。
> `@pytest.mark.agent` マーカーを付与する。
> LLM呼び出しはモックに置き換え、エージェントのフロー制御を確認する。

### テストケース一覧

| テストID | 対象 | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|------|---------|---------|---------------------|
| AGT-01 | スーパーバイザー | ツールを少なくとも1回選択してループが終了する | MockLLM を使用 | `trials_df` が 1 行以上になる |
| AGT-02 | スーパーバイザー | 試行回数の合計が `n_trials` を超えない | `n_trials=10` で実行 | `len(trials_df) <= 10` |
| AGT-03 | スーパーバイザー | `BayesianOptimizationTool` を選択できる | MockLLM が Bayesian を指示 | `trials_df["tool_used"]` に `"BayesianOptimizationTool"` が含まれる |
| AGT-04 | スーパーバイザー | `SobolSearchTool` を選択できる | MockLLM が Sobol を指示 | `trials_df["tool_used"]` に `"SobolSearchTool"` が含まれる |
| AGT-05 | スーパーバイザー | `ExpertAgentTool` を選択できる | MockLLM が Expert を指示 | `trials_df["tool_used"]` に `"ExpertAgentTool"` が含まれる |
| AGT-06 | ログ出力 | `logging.INFO` レベルでログが出力される | logging を INFO に設定して実行 | caplog に試行番号・スコア・ツール名が含まれる |
| AGT-07 | `supervisor_prompt` | スーパーバイザーに渡るプロンプトに反映される | `prompts={"supervisor": "テスト指示"}` | スーパーバイザーへの LLM 呼び出しで "テスト指示" が含まれる |
| AGT-08 | `expert_agent_prompt` | ExpertAgentTool 内エージェントに渡るプロンプトに反映される | `prompts={"expert_agent": "専門家指示"}` | ExpertAgentTool への LLM 呼び出しで "専門家指示" が含まれる |

---

## 5. 単体テスト：A-3 HPO性質テスト（`test_hpo_properties.py`）

> HPOとして期待される最適化の性質が成立するかを検証する。
> `@pytest.mark.hpo` マーカーを付与する。
> 軽量なダミーモデルと確定的な eval 関数を使い、実際にエージェントを動作させる。

### テストケース一覧

| テストID | 検証する HPO 性質 | 対象ツール / 機能 | 検証内容 | 許容誤差 |
|---------|----------------|----------------|---------|---------|
| HPO-01 | `best_score` が全試行の最良値と一致する | `HPOResult` | `best_score == trials_df["score"].max()` | 完全一致 |
| HPO-02 | `best_params` が `best_score` を達成した試行のパラメータと一致する | `HPOResult` | `best_params` を eval_fn に渡した結果が `best_score` に等しい | `atol=1e-6` |
| HPO-03 | ベイズ最適化が反復で改善傾向を示す（単調性は不要） | `BayesianOptimizationTool` | 確定的な2次関数 eval_fn で30試行後の best_score がランダム探索より高い | スコア比較 |
| HPO-04 | Sobol 探索が均一な探索空間をカバーする | `SobolSearchTool` | 探索したパラメータが探索空間全体に分布している（偏りがない） | 分位点ベース |
| HPO-05 | `trials_df` の行数が実際の試行回数と一致する | `HPOAgent.run` | `len(trials_df) == n_trials`（または n_trials 以下） | 完全一致 |
| HPO-06 | 同一条件で再実行してもクラッシュしない（安定性） | `HPOAgent.run` | 同じ引数で3回実行しても例外が発生しない | 例外なし |

---

## 6. 振る舞いテスト：グラフ出力による確認（`check_behavior.py`）

> 自動判定が困難な視覚的・定性的な性質を、グラフを出力して人間が確認する。
> 各グラフには「確認すべき観点」をタイトルまたは注釈として表示する。

### 確認項目一覧

| チェックID | 確認すべき観点 | 対応する仕様 | グラフの種類 | 合格の目安 |
|----------|-------------|------------|------------|----------|
| BHV-01 | スコア推移：試行を重ねるごとに best_score が改善しているか | HPO-03（ベイズ最適化の改善傾向） | 折れ線グラフ（x: trial_id, y: best_score の累積最大値） | 後半の試行ほど best_score が高い or 横ばいで安定している |
| BHV-02 | Sobol 探索の空間カバレッジ：探索点が2次元パラメータ空間に均等に分布しているか | HPO-04（Sobol の均一性） | 散布図（x: param_a, y: param_b） | 点がグリッド状・均等に広がっており、明らかな偏りがない |
| BHV-03 | ツール使用比率：スーパーバイザーがどのツールを何回選択したか | 要件定義 4.2（スーパーバイザーの動的判断） | 棒グラフ（x: ツール名, y: 使用回数） | 少なくとも1種類以上のツールが選ばれており、合計が n_trials に近い |
| BHV-04 | スコア分布：各ツールで達成されたスコアの分布に差があるか | 要件定義 4.3（各ツールの役割分担） | ボックスプロット（x: ツール名, y: score） | ツールごとに分布の特徴が確認できる（例：Bayesian が高スコアに集中） |
| BHV-05 | Markdown レポートの可読性：レポートが人間に読みやすい形式で出力されているか | 要件定義 6.2（テキストレポート） | ターミナルへの print 出力（グラフなし） | ヘッダー・最良パラメータ・考察・ツール内訳が含まれている |

---

## 7. 許容誤差の設定方針

数値比較を含むテストケースの許容誤差の設定根拠を明記する。

| テストID | 許容誤差 | 設定根拠 |
|---------|---------|---------|
| HPO-02 | `atol=1e-6` | float の eval_fn スコアの再現性。同一パラメータを渡せば完全一致するが、浮動小数点の演算誤差を考慮 |
| HPO-03 | スコア比較（厳密な閾値なし） | ベイズ最適化の優位性は確率的。30試行の平均で比較し、ランダム探索の best_score を上回ることを確認 |
| HPO-04 | 分位点ベース（各分位点に1点以上） | Sobol 列の理論的性質（低食違い量列）から、k分割した各区間に探索点が存在することを確認 |

---

## 8. テストディレクトリ構成

```
tests/
├── conftest.py              # MockLLM, DummyModel, DeterministicEvalFn の共通フィクスチャ
├── test_components.py       # A-1: コンポーネント実装テスト (@pytest.mark.component)
├── test_agent.py            # A-2: エージェント動作テスト (@pytest.mark.agent)
├── test_hpo_properties.py   # A-3: HPO性質テスト (@pytest.mark.hpo)
└── check_behavior.py        # B: 振る舞いテスト（グラフ出力、手動実行）
```

---

## 9. pytest設定（`pyproject.toml`）

```toml
[tool.pytest.ini_options]
markers = [
    "component: 各クラス・メソッドの実装正確性テスト",
    "agent: スーパーバイザーとツール群の連携・制御フローテスト",
    "hpo: HPOとして期待される最適化性質のテスト",
]
```

実行例：

```bash
# 全テスト実行
uv run pytest tests/

# サブカテゴリ別実行
uv run pytest tests/ -m component
uv run pytest tests/ -m agent
uv run pytest tests/ -m hpo

# 振る舞いテスト（手動確認）
uv run python tests/check_behavior.py
```

---

## 注意事項

- **LLM API は実テストで叩かない**：全テスト（A-1〜A-3）で LLM 呼び出しは必ずモックに置き換える。実際の API コストや非決定性をテストに持ち込まない
- **A-3 は軽量な eval_fn を使う**：DeterministicEvalFn は `params` の関数として確定的なスコアを返す単純な関数（例：2次関数）を使い、テスト速度を確保する
- **A-1・A-2・A-3 はファイル・マーカー・クラス名で明確に分ける**：1つのファイルに混在させない
- **振る舞いテスト（B）は自動判定しない**：`assert` を使わず、グラフタイトルに確認観点を表示して人間が目視確認する
- **許容誤差は根拠を明記する**：「なんとなく 1e-6」ではなく、設定理由を残す
- **`conftest.py` でフィクスチャを共通化する**：MockLLM・DummyModel・DeterministicEvalFn は全テストで再利用できるよう `conftest.py` に定義する
````