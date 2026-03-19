# テスト設計書：hpo-agent

## 1. テスト設計の方針

**参照ファイル**：`/doc/impl_design.md`、`/doc/requirements.md`、`/doc/llm_prompt_design.md`

LLM 呼び出しを含む全テスト（A-1〜A-3）では API キー不要・実行速度・コストを考慮して LLM を必ずモックに置き換える。HPO 性質テスト（A-3）では確定的な評価関数と軽量ダミーモデルを使い、乱数シードを固定することで再現可能なテストを実現する。`eval_duration` / `algo_duration` の計時が正しく記録されること、シードによる再現性が保証されることを重点的に検証する。

---

## 2. モック戦略

| モック名 | 対象 | 振る舞い | 使用するテスト |
|---------|------|---------|--------------|
| `MockSupervisorLLM` | Supervisor の LLM 呼び出し | ツール名と `n_trials` を含む固定の AIMessage を返す | A-1, A-2 |
| `MockExpertLLM` | ExpertAgentTool の LLM 呼び出し | `{"reasoning": "テスト提案", "params": {...}}` の固定 JSON を返す | A-1, A-2 |
| `DummyAdapter` | `ModelAdapterBase` 実装 | `evaluate()` が固定スコア（0.85）を返す。`time.sleep(0.001)` を入れ `eval_duration > 0` を保証 | A-1, A-2, A-3 |
| `DeterministicEvalFn` | eval 関数 | `-(num_leaves - 64)**2 / 1000 + 0.9` のような2次関数でスコアを確定的に返す | A-3（HPO 性質テスト） |
| `DummyLGBMModel` | LightGBM モデル | sklearn 互換の軽量クラス（`n_estimators=5`、`verbosity=-1`） | A-2 結合・A-3 |

---

## 3. 共通フィクスチャ（`conftest.py`）

```python
# tests/conftest.py

import pytest
from unittest.mock import MagicMock
from datetime import datetime
from hpo_agent.config import HPOConfig, ParamSpec, ParamSpace
from hpo_agent.records import TrialRecord


@pytest.fixture
def simple_param_space() -> ParamSpace:
    """テスト用の3パラメータ空間。"""
    return ParamSpace(specs=(
        ParamSpec(name="num_leaves", type="int", low=20, high=100),
        ParamSpec(name="learning_rate", type="float", low=0.01, high=0.3, log=True),
        ParamSpec(name="boosting_type", type="categorical", choices=("gbdt", "dart")),
    ))


@pytest.fixture
def sample_trial_records() -> list[TrialRecord]:
    """5件のサンプル試行履歴（スコア昇順）。"""
    return [
        TrialRecord(
            trial_id=i,
            params={"num_leaves": 20 + i * 10, "learning_rate": 0.05},
            score=0.70 + i * 0.05,
            tool_used="sobol_search",
            timestamp=datetime(2026, 3, 1, 0, i),
            eval_duration=0.5 + i * 0.1,
            algo_duration=0.01,
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_supervisor_llm() -> MagicMock:
    """Supervisor の LLM モック。sobol_search を n_trials=5 で1回実行して終了。"""
    llm = MagicMock()
    # 1回目：ツール呼び出し指示 → 2回目：終了（ツール呼び出しなし）
    llm.invoke.side_effect = [
        MagicMock(tool_calls=[{"name": "sobol_search", "args": {"n_trials": 5}}]),
        MagicMock(tool_calls=[]),
    ]
    return llm


@pytest.fixture
def mock_expert_llm() -> MagicMock:
    """ExpertAgentTool の LLM モック。正常 JSON を返す。"""
    llm = MagicMock()
    llm.invoke.return_value.content = (
        '{"reasoning": "テスト根拠", "params": {"num_leaves": 64, "learning_rate": 0.05}}'
    )
    return llm


@pytest.fixture
def dummy_adapter(simple_param_space) -> "DummyAdapter":
    from hpo_agent.adapters import ModelAdapterBase
    import time

    class DummyAdapter(ModelAdapterBase):
        def get_default_param_space(self) -> ParamSpace:
            return simple_param_space

        def evaluate(self, params: dict) -> float:
            time.sleep(0.001)  # eval_duration > 0 を保証
            return 0.85

    return DummyAdapter()


@pytest.fixture
def deterministic_eval_fn():
    """num_leaves=64 付近で最大スコアを持つ確定的な2次関数。"""
    def eval_fn(model, X, y) -> float:
        num_leaves = model.get_params().get("num_leaves", 50)
        return -(num_leaves - 64) ** 2 / 10000 + 0.9
    return eval_fn
```

---

## 4. 単体テスト：A-1 コンポーネント実装テスト（`test_components.py`）

> 各クラス・メソッドが仕様通りに動作するかを検証する。
> `@pytest.mark.component` マーカーを付与する。
> LLM 呼び出しはすべてモックに置き換える。

### テストケース一覧

| テスト ID | 対象クラス / メソッド | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|--------------------|---------|---------|--------------------|
| CMP-01 | `HPOAgent.__init__` | 必須引数で正常に初期化される | `model`, `eval_fn`, `n_trials` を渡す | 例外が発生しない |
| CMP-02 | `HPOAgent.__init__` | `seed=42` を渡すと `config.seed` に設定される | `seed=42` | `agent._config.seed == 42` |
| CMP-03 | `HPOAgent.__init__` | `seed` 省略時は `None` になる | `seed` を渡さない | `agent._config.seed is None` |
| CMP-04 | `HPOAgent.run` | 戻り値が `HPOResult` 型である | `MockSupervisorLLM` を使用して `run()` を呼ぶ | `isinstance(result, HPOResult)` が True |
| CMP-05 | `HPOResult.trials_df` | 必須カラムをすべて含む | `n_trials=5` で実行 | `trial_id`, `params`, `score`, `tool_used`, `timestamp`, `eval_duration`, `algo_duration`, `reasoning` カラムが存在する |
| CMP-06 | `HPOResult.report` | Markdown 形式の文字列が返る | `run()` を実行 | `#` または `##` を含む文字列が返る |
| CMP-07 | `HPOResult.report` | seed 値がレポートに含まれる | `seed=42` で実行 | レポートに `"42"` が含まれる |
| CMP-08 | `BayesianOptimizationTool._run` | 指定した試行回数の `TrialRecord` が返る | `n_trials=3`、`DummyAdapter` を使用 | `len(result) == 3` |
| CMP-09 | `BayesianOptimizationTool._run` | `algo_duration` が記録される | `n_trials=3` で実行 | 全 `TrialRecord` の `algo_duration >= 0` |
| CMP-10 | `SobolSearchTool._run` | 指定した試行回数の `TrialRecord` が返る | `n_trials=4`、`DummyAdapter` を使用 | `len(result) == 4` |
| CMP-11 | `SobolSearchTool._run` | `eval_duration` が計測される | `DummyAdapter`（`sleep(0.001)` あり）を使用 | 全 `TrialRecord` の `eval_duration > 0` |
| CMP-12 | `SobolSearchTool._run` | 生成パラメータが `ParamSpec` の範囲内 | `simple_param_space` で 20 件実行 | 全 `int` / `float` パラメータが `[low, high]` の範囲内 |
| CMP-13 | `SobolSearchTool._run` | `categorical` パラメータが choices の中から選ばれる | `simple_param_space` で 10 件実行 | `boosting_type` が `("gbdt", "dart")` のいずれか |
| CMP-14 | `ExpertAgentTool._run` | LLM から受け取ったパラメータで評価される | `MockExpertLLM` が `num_leaves=64` を返す | `TrialRecord.params["num_leaves"] == 64` |
| CMP-15 | `ExpertAgentTool._run` | `reasoning` が `TrialRecord` に格納される | `MockExpertLLM` が `reasoning` フィールドを含む JSON を返す | `TrialRecord.reasoning == "テスト根拠"` |
| CMP-16 | `ExpertAgentTool._run` | 不正 JSON で最大3回リトライし、3回目に成功 | 最初の2回は不正 JSON、3回目に正常 JSON | `len(result) == 1`、`llm.invoke.call_count == 3` |
| CMP-17 | `ExpertAgentTool._run` | 3回連続で不正 JSON の場合は例外送出 | 全回不正 JSON | `Exception` が発生する |
| CMP-18 | `ExpertAgentTool._run` | `algo_duration` に LLM 呼び出し時間が含まれる | `MockExpertLLM` で `n_trials=1` | `TrialRecord.algo_duration > 0` |
| CMP-19 | プロンプト結合 | デフォルト + ユーザープロンプトが連結される | `prompts={"supervisor": "追加指示"}` | Supervisor の LLM 呼び出しプロンプトが `"## ユーザー追加指示\n追加指示"` を含む |
| CMP-20 | プロンプト結合 | `prompts` が空の場合はデフォルトのみ | `prompts={}` | `"## ユーザー追加指示"` がプロンプトに含まれない |
| CMP-21 | `ParamSpec` | `log=True` かつ `low=0` の場合は `ValueError` | `ParamSpec(type="float", low=0.0, high=1.0, log=True)` | `ValueError` が発生する |
| CMP-22 | `ParamSpec` | `frozen=True` のためフィールドが変更不可 | インスタンス生成後にフィールドへ代入 | `FrozenInstanceError` が発生する |
| CMP-23 | `TrialRecord.to_dict` | `eval_duration` / `algo_duration` / `reasoning` が含まれる | `TrialRecord` を生成して `to_dict()` を呼ぶ | 辞書に 3 フィールドが存在する |
| CMP-24 | `LightGBMAdapter.evaluate` | 元モデルオブジェクトを変更しない | 評価前後で `model.get_params()` を比較 | 評価前後でパラメータが一致する（ディープコピーの確認） |
| CMP-25 | `HPOAgent._resolve_adapter` | ユーザー指定 `param_space` が優先される | `param_space=simple_param_space` | ツールに渡る `ParamSpace` が `simple_param_space` と一致する |
| CMP-26 | `HPOAgent._resolve_adapter` | 未対応モデル型で `TypeError` | `model=object()` を渡す | `TypeError` が発生する |

### テストコード骨格

```python
# tests/test_components.py

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.component


class TestHPOAgentInit:
    def test_seed_stored_in_config(self, lgbm_binary_setup):
        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y, seed=42)
        assert agent._config.seed == 42  # CMP-02

    def test_seed_default_none(self, lgbm_binary_setup):
        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        assert agent._config.seed is None  # CMP-03


class TestHPOResult:
    def test_trials_df_has_timing_columns(self, mock_hpo_result):
        assert "eval_duration" in mock_hpo_result.trials_df.columns  # CMP-05
        assert "algo_duration" in mock_hpo_result.trials_df.columns  # CMP-05


class TestExpertAgentToolRetry:
    def test_retry_succeeds_on_third_attempt(self, dummy_adapter, mock_expert_llm):
        mock_expert_llm.invoke.side_effect = [
            MagicMock(content="invalid json"),
            MagicMock(content="also invalid"),
            MagicMock(content='{"reasoning": "ok", "params": {"num_leaves": 64, "learning_rate": 0.1}}'),
        ]
        tool = ExpertAgentTool(adapter=dummy_adapter, llm=mock_expert_llm, system_prompt="test")
        results = tool._run(n_trials=1, trial_history=[])
        assert len(results) == 1
        assert mock_expert_llm.invoke.call_count == 3  # CMP-16

    def test_all_retries_fail_raises(self, dummy_adapter, mock_expert_llm):
        mock_expert_llm.invoke.return_value.content = "not json"
        tool = ExpertAgentTool(adapter=dummy_adapter, llm=mock_expert_llm, system_prompt="test")
        with pytest.raises(Exception):
            tool._run(n_trials=1, trial_history=[])  # CMP-17
```

---

## 5. 単体テスト：A-2 エージェント動作テスト（`test_agent.py`）

> スーパーバイザーとツール群の連携・LangGraph の制御フローを検証する。
> `@pytest.mark.agent` マーカーを付与する。
> LLM 呼び出しはモックに置き換え、エージェントのフロー制御を確認する。

### テストケース一覧

| テスト ID | 対象 | 検証内容 | 入力条件 | 期待される出力 / 状態 |
|---------|------|---------|---------|---------------------|
| AGT-01 | スーパーバイザー | ツールを少なくとも1回選択してループが終了する | `MockSupervisorLLM` を使用 | `trials_df` が1行以上になる |
| AGT-02 | スーパーバイザー | 試行回数の合計が `n_trials` を超えない | `n_trials=10` で実行 | `len(trials_df) <= 10` |
| AGT-03 | スーパーバイザー | `SobolSearchTool` を選択できる | `MockSupervisorLLM` が Sobol を指示 | `trials_df["tool_used"]` に `"sobol_search"` が含まれる |
| AGT-04 | スーパーバイザー | `BayesianOptimizationTool` を選択できる | `MockSupervisorLLM` が Bayesian を指示 | `trials_df["tool_used"]` に `"bayesian_optimization"` が含まれる |
| AGT-05 | スーパーバイザー | `ExpertAgentTool` を選択できる | `MockSupervisorLLM` が Expert を指示 | `trials_df["tool_used"]` に `"expert_agent"` が含まれる |
| AGT-06 | 中間レポート | ツール完了ごとに `logging.INFO` が出力される | `logging` を INFO に設定して実行 | `caplog` に `"sobol_search"` 等のツール名が含まれる |
| AGT-07 | 中間レポート | 中間レポートに実行時間が含まれる | `DummyAdapter`（sleep あり）を使用 | caplog に時間関連の文字列が含まれる |
| AGT-08 | `supervisor_prompt` | スーパーバイザーへの LLM 呼び出しに追加プロンプトが含まれる | `prompts={"supervisor": "テスト指示"}` | LLM に渡された引数に `"テスト指示"` が含まれる |
| AGT-09 | `expert_agent_prompt` | ExpertAgentTool の LLM 呼び出しに追加プロンプトが含まれる | `prompts={"expert_agent": "専門家指示"}` | ExpertAgentTool の LLM 呼び出しに `"専門家指示"` が含まれる |
| AGT-10 | 試行履歴の引き継ぎ | 2回目以降のツール呼び出しに前回の試行履歴が渡される | 複数ツールを連続実行 | 2回目のツール呼び出し時の `trial_history` が1回目の結果を含む |
| AGT-11 | シード伝播 | `seed=42` が `SobolSearchTool` に伝播される | `seed=42` で実行 | `SobolSearchTool` のインスタンスが `seed=42` で初期化されている |
| AGT-12 | シード伝播 | `seed=42` が `BayesianOptimizationTool` に伝播される | `seed=42` で実行 | Optuna の `TPESampler` が `seed=42` で生成されている |
| AGT-13 | `ExpertAgentTool` 履歴選択 | 30件の履歴からスコア上位20件＋直近10件が選ばれる | `trial_history` が 30 件ある状態で実行 | LLM に渡された JSON の `trial_id` が重複なしで最大 30 件以内 |

### テストコード骨格

```python
# tests/test_agent.py

import pytest
import logging

pytestmark = pytest.mark.agent


class TestSupervisorLoop:
    def test_n_trials_not_exceeded(self, mock_supervisor_llm, dummy_adapter):
        # MockSupervisorLLM が n_trials=12 を要求しても n_trials=10 で上限適用
        result = run_with_mock(n_trials=10, llm=mock_supervisor_llm, adapter=dummy_adapter)
        assert len(result.trials_df) <= 10  # AGT-02

    def test_intermediate_report_logged(self, mock_supervisor_llm, dummy_adapter, caplog):
        with caplog.at_level(logging.INFO):
            run_with_mock(n_trials=5, llm=mock_supervisor_llm, adapter=dummy_adapter)
        assert any("sobol_search" in record.message for record in caplog.records)  # AGT-06


class TestSeedPropagation:
    def test_seed_propagated_to_sobol(self, mocker, mock_supervisor_llm, dummy_adapter):
        sobol_init = mocker.patch("hpo_agent.tools.sobol.SobolSearchTool.__init__",
                                  wraps=SobolSearchTool.__init__)
        run_with_mock(n_trials=5, llm=mock_supervisor_llm, adapter=dummy_adapter, seed=42)
        call_kwargs = sobol_init.call_args.kwargs
        assert call_kwargs.get("seed") == 42  # AGT-11
```

---

## 6. 単体テスト：A-3 HPO 性質テスト（`test_hpo_properties.py`）

> HPO として期待される最適化の性質が成立するかを検証する。
> `@pytest.mark.hpo` マーカーを付与する。
> 軽量なダミーモデルと確定的な eval 関数を使い、実際にエージェントを動作させる。

### テストケース一覧

| テスト ID | 検証する HPO 性質 | 対象ツール / 機能 | 検証内容 | 許容誤差 |
|---------|----------------|----------------|---------|---------|
| HPO-01 | `best_score` が全試行の最良値と一致する | `HPOResult` | `best_score == trials_df["score"].max()` | 完全一致 |
| HPO-02 | `best_params` が `best_score` を達成したパラメータと一致する | `HPOResult` | `eval_fn(best_params)` の結果が `best_score` に等しい | `atol=1e-6` |
| HPO-03 | ベイズ最適化が同一シードで再現可能 | `BayesianOptimizationTool` | `seed=42` で2回実行し、全 `TrialRecord` のパラメータが一致 | 完全一致 |
| HPO-04 | Sobol 探索が同一シードで再現可能 | `SobolSearchTool` | `seed=42` で2回実行し、全 `TrialRecord` のパラメータが一致 | 完全一致 |
| HPO-05 | Sobol 探索がパラメータ空間を均等にカバーする | `SobolSearchTool` | 探索点を k 分割した各区間に少なくとも1点存在する | 分位点ベース（k=4） |
| HPO-06 | ベイズ最適化がランダム探索より高いスコアを達成する傾向がある | `BayesianOptimizationTool` | 確定的 eval_fn で30試行後の `best_score` がランダム探索の `best_score` を上回る | スコア比較 |
| HPO-07 | `trials_df` の行数が総試行回数以内 | `HPOAgent.run` | `len(trials_df) <= n_trials` | 完全一致 |
| HPO-08 | 全試行の `eval_duration` が正の値 | 計時機能 | `trials_df["eval_duration"].min() > 0` | 完全一致 |
| HPO-09 | 全試行の `algo_duration` が非負 | 計時機能 | `trials_df["algo_duration"].min() >= 0` | 完全一致 |
| HPO-10 | 同一条件で3回実行してもクラッシュしない（安定性） | `HPOAgent.run` | 同じ引数で3回実行しても例外が発生しない | 例外なし |

### テストコード骨格

```python
# tests/test_hpo_properties.py

import pytest
import numpy as np

pytestmark = pytest.mark.hpo


class TestResultCorrectness:
    def test_best_score_equals_max(self, hpo_result_from_sobol):
        result = hpo_result_from_sobol
        assert result.best_score == result.trials_df["score"].max()  # HPO-01

    def test_best_params_achieves_best_score(self, hpo_result_from_sobol, deterministic_eval_fn):
        result = hpo_result_from_sobol
        achieved = deterministic_eval_fn_call(result.best_params)
        assert abs(achieved - result.best_score) < 1e-6  # HPO-02


class TestSeedReproducibility:
    def test_sobol_reproducible_with_seed(self, dummy_adapter, simple_param_space):
        tool_a = SobolSearchTool(adapter=dummy_adapter, param_space=simple_param_space, seed=42)
        tool_b = SobolSearchTool(adapter=dummy_adapter, param_space=simple_param_space, seed=42)
        results_a = tool_a._run(n_trials=10, trial_history=[])
        results_b = tool_b._run(n_trials=10, trial_history=[])
        assert [r.params for r in results_a] == [r.params for r in results_b]  # HPO-04

    def test_bayesian_reproducible_with_seed(self, dummy_adapter, simple_param_space):
        tool_a = BayesianOptimizationTool(adapter=dummy_adapter, param_space=simple_param_space, seed=42)
        tool_b = BayesianOptimizationTool(adapter=dummy_adapter, param_space=simple_param_space, seed=42)
        history = []  # 同一の履歴で開始
        results_a = tool_a._run(n_trials=5, trial_history=history)
        results_b = tool_b._run(n_trials=5, trial_history=history)
        assert [r.params for r in results_a] == [r.params for r in results_b]  # HPO-03


class TestSobolCoverage:
    def test_sobol_covers_param_space_uniformly(self, dummy_adapter, simple_param_space):
        """num_leaves（int 型）を 4 分位に分割し、各区間に探索点があることを確認。"""
        tool = SobolSearchTool(adapter=dummy_adapter, param_space=simple_param_space, seed=0)
        results = tool._run(n_trials=20, trial_history=[])
        values = [r.params["num_leaves"] for r in results]
        # [20, 40), [40, 60), [60, 80), [80, 100] の4区間に少なくとも1点
        bins = [20, 40, 60, 80, 100]
        for lo, hi in zip(bins[:-1], bins[1:]):
            assert any(lo <= v <= hi for v in values), f"区間 [{lo}, {hi}] に探索点がない"  # HPO-05


class TestTimingMeasurement:
    def test_eval_duration_positive(self, sobol_results):
        assert all(r.eval_duration > 0 for r in sobol_results)  # HPO-08

    def test_algo_duration_non_negative(self, sobol_results):
        assert all(r.algo_duration >= 0 for r in sobol_results)  # HPO-09
```

---

## 7. 振る舞いテスト：グラフ出力による確認（`check_behavior.py`）

> 自動判定が困難な視覚的・定性的な性質を、グラフを出力して人間が確認する。
> 各グラフには「確認すべき観点」をタイトルまたは注釈として表示する。

### 確認項目一覧

| チェック ID | 確認すべき観点 | 対応する仕様 | グラフの種類 | 合格の目安 |
|----------|-------------|------------|------------|----------|
| BHV-01 | スコア推移：試行を重ねるごとに `best_score` が改善しているか | HPO-06（ベイズ最適化の改善傾向） | 折れ線グラフ（x: trial_id, y: 累積最大スコア） | 後半ほど `best_score` が高い、または横ばいで安定している |
| BHV-02 | Sobol 探索の空間カバレッジ：探索点が2次元空間に均等に分布しているか | HPO-05（Sobol の均一性） | 散布図（x: num_leaves, y: learning_rate） | 点がグリッド状・均等に広がり、明らかな偏りがない |
| BHV-03 | ツール使用比率：スーパーバイザーがどのツールを何回選択したか | 要件定義 §4.2（スーパーバイザーの動的判断） | 棒グラフ（x: ツール名, y: 使用回数） | 少なくとも1種類以上のツールが選ばれており、合計が `n_trials` に近い |
| BHV-04 | スコア分布：各ツールで達成されたスコアの分布に差があるか | 要件定義 §4.3（各ツールの役割分担） | ボックスプロット（x: ツール名, y: score） | ツールごとに分布の特徴が確認できる |
| BHV-05 | 実行時間の内訳：`eval_duration` と `algo_duration` の比率が合理的か | 要件定義 §6.1（計時カラム） | 積み上げ棒グラフ（x: ツール名, y: 平均 eval_duration / algo_duration） | `eval_duration` がモデル学習時間として支配的であることが目視で確認できる |
| BHV-06 | Markdown レポートの可読性：レポートが人間に読みやすい形式で出力されているか | 要件定義 §6.2（テキストレポート） | ターミナルへの print 出力（グラフなし） | ヘッダー・最良パラメータ・時間サマリー・seed 情報が含まれている |

### スクリプト骨格

```python
# tests/check_behavior.py
"""
振る舞いテスト：グラフ出力による定性確認スクリプト。
実行方法: uv run python tests/check_behavior.py
"""

import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from hpo_agent import HPOAgent


def make_env():
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    model = lgb.LGBMClassifier(verbosity=-1)
    def eval_fn(m, X, y): return accuracy_score(y, m.predict(X))
    return model, eval_fn, X, y


def check_bhv01_score_trend(result):
    """BHV-01: スコア推移グラフ。改善傾向が見られるか確認。"""
    df = result.trials_df.sort_values("trial_id")
    cummax = df["score"].cummax()
    plt.figure()
    plt.plot(df["trial_id"], cummax, marker="o")
    plt.title("BHV-01 スコア推移（累積最大値）\n確認観点: 後半ほど高い or 安定して横ばい")
    plt.xlabel("trial_id"); plt.ylabel("best_score (cummax)")
    plt.tight_layout(); plt.show()


def check_bhv02_sobol_coverage(result):
    """BHV-02: Sobol 探索の空間カバレッジ散布図。均等な分布を確認。"""
    sobol_df = result.trials_df[result.trials_df["tool_used"] == "sobol_search"]
    num_leaves = [p["num_leaves"] for p in sobol_df["params"]]
    lr = [p["learning_rate"] for p in sobol_df["params"]]
    plt.figure()
    plt.scatter(num_leaves, lr)
    plt.title("BHV-02 Sobol 探索のカバレッジ\n確認観点: 点が均等に広がっているか")
    plt.xlabel("num_leaves"); plt.ylabel("learning_rate")
    plt.tight_layout(); plt.show()


def check_bhv05_timing_breakdown(result):
    """BHV-05: ツール別実行時間内訳。eval_duration と algo_duration の比率を確認。"""
    df = result.trials_df
    grouped = df.groupby("tool_used")[["eval_duration", "algo_duration"]].mean()
    grouped.plot(kind="bar", stacked=True)
    plt.title("BHV-05 ツール別平均実行時間\n確認観点: eval_duration が支配的か")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    model, eval_fn, X, y = make_env()
    agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=30, X=X, y=y, seed=42)
    result = agent.run()

    print("\n=== BHV-06: Markdown レポート ===")
    print(result.report)

    check_bhv01_score_trend(result)
    check_bhv02_sobol_coverage(result)
    check_bhv05_timing_breakdown(result)
```

---

## 8. 許容誤差の設定方針

| テスト ID | 許容誤差 | 設定根拠 |
|---------|---------|---------|
| HPO-02 | `atol=1e-6` | 同一パラメータを eval_fn に渡せば原則一致するが、浮動小数点演算誤差を考慮 |
| HPO-03, HPO-04 | 完全一致 | 同一シード・同一アルゴリズムでは決定的に同一パラメータが生成される |
| HPO-05 | 分位点ベース（k=4 区間） | Sobol 列の理論的性質（低食違い量列）から各区間への網羅を期待する |
| HPO-06 | スコア比較（厳密な閾値なし） | ベイズ最適化の優位性は確率的。確定的 eval_fn で 30 試行後の `best_score` がランダム探索を上回ることを確認 |

---

## 9. テストディレクトリ構成

```
tests/
├── conftest.py              # 共通フィクスチャ（MockLLM・DummyAdapter・DeterministicEvalFn 等）
├── test_components.py       # A-1: コンポーネント実装テスト (@pytest.mark.component)
├── test_agent.py            # A-2: エージェント動作テスト (@pytest.mark.agent)
├── test_hpo_properties.py   # A-3: HPO 性質テスト (@pytest.mark.hpo)
└── check_behavior.py        # B: 振る舞いテスト（グラフ出力・手動実行）
```

---

## 10. pytest 設定（`pyproject.toml`）

```toml
[tool.pytest.ini_options]
markers = [
    "component: 各クラス・メソッドの実装正確性テスト",
    "agent: スーパーバイザーとツール群の連携・制御フローテスト",
    "hpo: HPO として期待される最適化性質のテスト",
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

## 11. 注意事項

- **LLM API は自動テストで叩かない**：全テスト（A-1〜A-3）で LLM 呼び出しはモックに置き換える。実際の API コストや非決定性をテストに持ち込まない
- **A-3 は軽量な `DeterministicEvalFn` を使う**：`-(num_leaves - 64)^2 / 10000 + 0.9` のような2次関数で、テスト速度と決定論を確保する
- **計時テストの安定性**：`DummyAdapter.evaluate()` に `time.sleep(0.001)` を入れ、`eval_duration > 0` を確実に保証する
- **seed の持ち越しを防ぐ**：HPO-03・HPO-04 ではツールインスタンスを毎回新規生成し、内部乱数状態が持ち越されないことを確認する
- **振る舞いテスト（B）は自動判定しない**：`assert` を使わず、グラフタイトルに確認観点を表示して人間が目視確認する
- **`conftest.py` でフィクスチャを共通化する**：全テストファイルで再利用できるよう `conftest.py` に定義する
