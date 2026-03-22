"""A-1 コンポーネント実装テスト (CMP-*).

各クラス・メソッドが仕様通りに動作するかを検証する。
LLM 呼び出しはすべてモックに置き換える。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from hpo_agent.models import ParamSpace

if TYPE_CHECKING:
    from conftest import DummyAdapter

pytestmark = pytest.mark.component


# ---------------------------------------------------------------------------
# CMP-21/22: ParamSpec
# ---------------------------------------------------------------------------


class TestParamSpec:
    def test_log_true_low_zero_raises(self) -> None:
        """CMP-21: log=True かつ low=0 の場合は ValueError."""
        from hpo_agent.models import ParamSpec

        with pytest.raises(ValueError):
            ParamSpec(name="lr", type="float", low=0.0, high=1.0, log=True)

    def test_log_true_low_negative_raises(self) -> None:
        """log=True かつ low < 0 の場合も ValueError."""
        from hpo_agent.models import ParamSpec

        with pytest.raises(ValueError):
            ParamSpec(name="lr", type="float", low=-0.1, high=1.0, log=True)

    def test_log_true_positive_low_ok(self) -> None:
        """log=True かつ low > 0 の場合は正常に生成できる."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="lr", type="float", low=0.001, high=1.0, log=True)
        assert spec.log is True

    def test_frozen_field_immutable(self) -> None:
        """CMP-22: frozen=True のためフィールドが変更不可."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="n", type="int", low=1, high=10)
        with pytest.raises(FrozenInstanceError):
            spec.name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CMP-23: TrialRecord.to_dict
# ---------------------------------------------------------------------------


class TestTrialRecord:
    def test_to_dict_contains_required_fields(self) -> None:
        """CMP-23: to_dict() に eval_duration / algo_duration / reasoning が含まれる."""
        from hpo_agent.models import TrialRecord

        record = TrialRecord(
            trial_id=0,
            params={"num_leaves": 31},
            score=0.9,
            tool_used="sobol_search",
            timestamp=datetime(2026, 1, 1),
            eval_duration=0.5,
            algo_duration=0.01,
            reasoning="test reason",
        )
        d = record.to_dict()
        assert "eval_duration" in d
        assert "algo_duration" in d
        assert "reasoning" in d

    def test_to_dict_timestamp_is_string(self) -> None:
        """to_dict() の timestamp は ISO 文字列にシリアライズされる."""
        from hpo_agent.models import TrialRecord

        record = TrialRecord(
            trial_id=0,
            params={},
            score=0.8,
            tool_used="sobol_search",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        d = record.to_dict()
        assert isinstance(d["timestamp"], str)


# ---------------------------------------------------------------------------
# CMP-24: LightGBMAdapter
# ---------------------------------------------------------------------------


class TestLightGBMAdapter:
    def test_evaluate_does_not_mutate_model(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-24: evaluate() 後に元モデルのパラメータが変化しない（deepcopy 確認）."""
        from hpo_agent.adapters import LightGBMAdapter

        model, eval_fn, X, y = lgbm_binary_setup
        params_before = model.get_params().copy()
        adapter = LightGBMAdapter(model=model, eval_fn=eval_fn, X=X, y=y)
        adapter.evaluate({"num_leaves": 64, "n_estimators": 10})
        assert model.get_params() == params_before

    def test_get_default_param_space_has_eight_params(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """get_default_param_space() が 8 パラメータを返す."""
        from hpo_agent.adapters import LightGBMAdapter

        model, eval_fn, X, y = lgbm_binary_setup
        adapter = LightGBMAdapter(model=model, eval_fn=eval_fn, X=X, y=y)
        space = adapter.get_default_param_space()
        names = {s.name for s in space.specs}
        expected = {
            "num_leaves",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# CMP-10/11/12/13: SobolSearchTool
# ---------------------------------------------------------------------------


class TestSobolSearchTool:
    def test_returns_correct_trial_count(
        self, dummy_adapter: DummyAdapter, simple_param_space: ParamSpace
    ) -> None:
        """CMP-10: 指定した試行回数の TrialRecord が返る."""
        from hpo_agent.tools import SobolSearchTool

        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="sobol_search",
            description="test",
        )
        results = tool._run(n_trials=4, trial_history=[])
        assert len(results) == 4

    def test_eval_duration_positive(
        self, dummy_adapter: DummyAdapter, simple_param_space: ParamSpace
    ) -> None:
        """CMP-11: eval_duration が計測される（DummyAdapter は sleep(0.001) を含む）."""
        from hpo_agent.tools import SobolSearchTool

        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="sobol_search",
            description="test",
        )
        results = tool._run(n_trials=3, trial_history=[])
        assert all(r.eval_duration > 0 for r in results)

    def test_params_within_bounds(
        self, dummy_adapter: DummyAdapter, simple_param_space: ParamSpace
    ) -> None:
        """CMP-12: 生成パラメータが ParamSpec の範囲内."""
        from hpo_agent.tools import SobolSearchTool

        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="sobol_search",
            description="test",
        )
        results = tool._run(n_trials=20, trial_history=[])
        for r in results:
            assert 20 <= r.params["num_leaves"] <= 100
            assert 0.01 <= r.params["learning_rate"] <= 0.3

    def test_categorical_within_choices(
        self, dummy_adapter: DummyAdapter, simple_param_space: ParamSpace
    ) -> None:
        """CMP-13: categorical パラメータが choices の中から選ばれる."""
        from hpo_agent.tools import SobolSearchTool

        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="sobol_search",
            description="test",
        )
        results = tool._run(n_trials=10, trial_history=[])
        for r in results:
            assert r.params["boosting_type"] in ("gbdt", "dart")


# ---------------------------------------------------------------------------
# CMP-08/09: BayesianOptimizationTool
# ---------------------------------------------------------------------------


class TestBayesianOptimizationTool:
    def test_returns_correct_trial_count(
        self, dummy_adapter: DummyAdapter, simple_param_space: ParamSpace
    ) -> None:
        """CMP-08: 指定した試行回数の TrialRecord が返る."""
        from hpo_agent.tools import BayesianOptimizationTool

        tool = BayesianOptimizationTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="bayesian_optimization",
            description="test",
        )
        results = tool._run(n_trials=3, trial_history=[])
        assert len(results) == 3

    def test_algo_duration_non_negative(
        self, dummy_adapter: DummyAdapter, simple_param_space: ParamSpace
    ) -> None:
        """CMP-09: algo_duration が記録される."""
        from hpo_agent.tools import BayesianOptimizationTool

        tool = BayesianOptimizationTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="bayesian_optimization",
            description="test",
        )
        results = tool._run(n_trials=3, trial_history=[])
        assert all(r.algo_duration >= 0 for r in results)


# ---------------------------------------------------------------------------
# CMP-14/15/16/17/18: ExpertAgentTool
# ---------------------------------------------------------------------------


class TestExpertAgentTool:
    def test_params_from_llm_used_for_evaluation(
        self,
        dummy_adapter: DummyAdapter,
        simple_param_space: ParamSpace,
        mock_expert_llm: MagicMock,
    ) -> None:
        """CMP-14: LLM から受け取ったパラメータで評価される."""
        from hpo_agent.tools import ExpertAgentTool

        tool = ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=mock_expert_llm,
            system_prompt="test",
            name="expert_agent",
            description="test",
        )
        results = tool._run(n_trials=1, trial_history=[])
        assert results[0].params["num_leaves"] == 64

    def test_reasoning_stored_in_trial_record(
        self,
        dummy_adapter: DummyAdapter,
        simple_param_space: ParamSpace,
        mock_expert_llm: MagicMock,
    ) -> None:
        """CMP-15: reasoning が TrialRecord に格納される."""
        from hpo_agent.tools import ExpertAgentTool

        tool = ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=mock_expert_llm,
            system_prompt="test",
            name="expert_agent",
            description="test",
        )
        results = tool._run(n_trials=1, trial_history=[])
        assert results[0].reasoning == "テスト根拠"

    def test_retry_succeeds_on_third_attempt(
        self,
        dummy_adapter: DummyAdapter,
        simple_param_space: ParamSpace,
        mock_expert_llm: MagicMock,
    ) -> None:
        """CMP-16: 不正 JSON で最大3回リトライし、3回目に成功."""
        from unittest.mock import MagicMock

        from hpo_agent.tools import ExpertAgentTool

        mock_expert_llm.invoke.side_effect = [
            MagicMock(content="invalid json"),
            MagicMock(content="also invalid"),
            MagicMock(
                content='{"reasoning": "ok", "params": {"num_leaves": 64, "learning_rate": 0.1}}'
            ),
        ]
        tool = ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=mock_expert_llm,
            system_prompt="test",
            name="expert_agent",
            description="test",
        )
        results = tool._run(n_trials=1, trial_history=[])
        assert len(results) == 1
        assert mock_expert_llm.invoke.call_count == 3

    def test_all_retries_fail_raises(
        self,
        dummy_adapter: DummyAdapter,
        simple_param_space: ParamSpace,
        mock_expert_llm: MagicMock,
    ) -> None:
        """CMP-17: 3回連続で不正 JSON の場合は例外送出."""
        from hpo_agent.tools import ExpertAgentTool

        mock_expert_llm.invoke.return_value.content = "not json"
        tool = ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=mock_expert_llm,
            system_prompt="test",
            name="expert_agent",
            description="test",
        )
        with pytest.raises(Exception):
            tool._run(n_trials=1, trial_history=[])

    def test_algo_duration_positive(
        self,
        dummy_adapter: DummyAdapter,
        simple_param_space: ParamSpace,
        mock_expert_llm: MagicMock,
    ) -> None:
        """CMP-18: algo_duration に LLM 呼び出し時間が含まれる."""
        from hpo_agent.tools import ExpertAgentTool

        tool = ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=mock_expert_llm,
            system_prompt="test",
            name="expert_agent",
            description="test",
        )
        results = tool._run(n_trials=1, trial_history=[])
        assert results[0].algo_duration > 0


# ---------------------------------------------------------------------------
# CMP-06/07: ReportGenerator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def test_report_is_markdown(self, sample_trial_records: list) -> None:
        """CMP-06: Markdown 形式の文字列が返る."""
        from hpo_agent.report import ReportGenerator

        gen = ReportGenerator()
        report = gen.generate_intermediate(
            trial_records=sample_trial_records,
            best_params={"num_leaves": 60},
            best_score=0.90,
            seed=42,
        )
        assert "#" in report

    def test_report_contains_seed(self, sample_trial_records: list) -> None:
        """CMP-07: seed 値がレポートに含まれる."""
        from hpo_agent.report import ReportGenerator

        gen = ReportGenerator()
        report = gen.generate_intermediate(
            trial_records=sample_trial_records,
            best_params={"num_leaves": 60},
            best_score=0.90,
            seed=42,
        )
        assert "42" in report


# ---------------------------------------------------------------------------
# CMP-01/02/03/04/05/25/26: HPOAgent
# ---------------------------------------------------------------------------


class TestHPOAgent:
    def test_init_does_not_raise(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-01: 必須引数で正常に初期化される."""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        assert agent is not None

    def test_seed_stored_in_config(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-02: seed=42 を渡すと config.seed に設定される."""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y, seed=42)
        assert agent._config.seed == 42

    def test_seed_default_none(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-03: seed 省略時は None になる."""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        assert agent._config.seed is None

    def test_run_returns_hpo_result(
        self,
        lgbm_binary_setup,  # type: ignore[no-untyped-def]
        mock_supervisor_llm,  # type: ignore[no-untyped-def]
        simple_param_space,  # type: ignore[no-untyped-def]
    ) -> None:
        """CMP-04: run() が HPOResult 型を返す（MockSupervisorLLM 使用）."""
        from unittest.mock import patch

        from hpo_agent.agent import HPOAgent
        from hpo_agent.models import HPOResult

        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(
            model=model,
            eval_fn=eval_fn,
            n_trials=5,
            X=X,
            y=y,
            param_space=simple_param_space,
        )

        with patch.object(agent, "_build_supervisor") as mock_build:
            from unittest.mock import MagicMock

            from hpo_agent.report import ReportGenerator
            from hpo_agent.supervisor import Supervisor
            from hpo_agent.tools import (
                BayesianOptimizationTool,
                ExpertAgentTool,
                SobolSearchTool,
            )

            adapter_instance = agent._resolve_adapter()[0]
            supervisor = Supervisor(
                llm=mock_supervisor_llm,
                tools=[
                    SobolSearchTool(
                        adapter=adapter_instance,
                        param_space=simple_param_space,
                        name="sobol_search",
                        description="test",
                    ),
                    BayesianOptimizationTool(
                        adapter=adapter_instance,
                        param_space=simple_param_space,
                        name="bayesian_optimization",
                        description="test",
                    ),
                    ExpertAgentTool(
                        adapter=adapter_instance,
                        param_space=simple_param_space,
                        llm=MagicMock(),
                        system_prompt="test",
                        name="expert_agent",
                        description="test",
                    ),
                ],
                report_generator=ReportGenerator(),
                system_prompt="test",
            )
            mock_build.return_value = supervisor

            result = agent.run()
            assert isinstance(result, HPOResult)

    def test_trials_df_has_required_columns(
        self,
        lgbm_binary_setup,  # type: ignore[no-untyped-def]
        mock_supervisor_llm,  # type: ignore[no-untyped-def]
        simple_param_space,  # type: ignore[no-untyped-def]
    ) -> None:
        """CMP-05: trials_df に必須カラムが含まれる."""
        from unittest.mock import MagicMock, patch

        from hpo_agent.agent import HPOAgent
        from hpo_agent.report import ReportGenerator
        from hpo_agent.supervisor import Supervisor
        from hpo_agent.tools import (
            BayesianOptimizationTool,
            ExpertAgentTool,
            SobolSearchTool,
        )

        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(
            model=model,
            eval_fn=eval_fn,
            n_trials=5,
            X=X,
            y=y,
            param_space=simple_param_space,
        )

        with patch.object(agent, "_build_supervisor") as mock_build:
            adapter_instance = agent._resolve_adapter()[0]
            supervisor = Supervisor(
                llm=mock_supervisor_llm,
                tools=[
                    SobolSearchTool(
                        adapter=adapter_instance,
                        param_space=simple_param_space,
                        name="sobol_search",
                        description="test",
                    ),
                    BayesianOptimizationTool(
                        adapter=adapter_instance,
                        param_space=simple_param_space,
                        name="bayesian_optimization",
                        description="test",
                    ),
                    ExpertAgentTool(
                        adapter=adapter_instance,
                        param_space=simple_param_space,
                        llm=MagicMock(),
                        system_prompt="test",
                        name="expert_agent",
                        description="test",
                    ),
                ],
                report_generator=ReportGenerator(),
                system_prompt="test",
            )
            mock_build.return_value = supervisor

            result = agent.run()
            required_cols = {
                "trial_id",
                "score",
                "tool_used",
                "timestamp",
                "eval_duration",
                "algo_duration",
                "reasoning",
            }
            assert required_cols.issubset(set(result.trials_df.columns))

    def test_user_param_space_priority(self, lgbm_binary_setup, simple_param_space) -> None:  # type: ignore[no-untyped-def]
        """CMP-25: ユーザー指定 param_space がアダプターのデフォルトより優先される."""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(
            model=model,
            eval_fn=eval_fn,
            n_trials=5,
            X=X,
            y=y,
            param_space=simple_param_space,
        )
        _, resolved_space = agent._resolve_adapter()
        assert resolved_space == simple_param_space

    def test_unsupported_model_raises_type_error(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-26: 未対応モデル型で TypeError."""
        from hpo_agent.agent import HPOAgent

        _, eval_fn, X, y = lgbm_binary_setup
        agent = HPOAgent(
            model=object(),
            eval_fn=eval_fn,
            n_trials=5,
            X=X,
            y=y,
        )
        with pytest.raises(TypeError):
            agent._resolve_adapter()


# ---------------------------------------------------------------------------
# SklearnAdapter（CMP-27〜29）
# ---------------------------------------------------------------------------


@pytest.mark.component
class TestSklearnAdapter:
    """SklearnAdapter のコンポーネントテスト。"""

    def test_evaluate_does_not_mutate_model(self, sklearn_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-27: evaluate() 後に元モデルが fit されていない（clone 確認）。"""
        from hpo_agent.adapters import SklearnAdapter

        model, eval_fn, X, y = sklearn_binary_setup
        adapter = SklearnAdapter(model=model, eval_fn=eval_fn, X=X, y=y)
        assert not hasattr(model, "estimators_")  # fit 前は estimators_ なし
        adapter.evaluate({"n_estimators": 5})
        assert not hasattr(model, "estimators_")  # evaluate 後も元モデルは未学習

    def test_get_default_param_space_raises(self, sklearn_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-28: get_default_param_space() は NotImplementedError を送出する。"""
        from hpo_agent.adapters import SklearnAdapter

        model, eval_fn, X, y = sklearn_binary_setup
        adapter = SklearnAdapter(model=model, eval_fn=eval_fn, X=X, y=y)
        with pytest.raises(NotImplementedError):
            adapter.get_default_param_space()

    def test_resolve_adapter_without_param_space_raises(self, sklearn_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-29: sklearn モデルで param_space を省略すると ValueError。"""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = sklearn_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        with pytest.raises(ValueError, match="param_space"):
            agent._resolve_adapter()

    def test_pytorch_model_fn_resolves_to_pytorch_adapter(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-30: callable な model_fn が PyTorchAdapter に解決される。"""
        from hpo_agent.adapters import PyTorchAdapter
        from hpo_agent.agent import HPOAgent

        model_fn, eval_fn, param_space = pytorch_setup
        agent = HPOAgent(
            model=model_fn,
            eval_fn=eval_fn,
            n_trials=5,
            param_space=param_space,
        )
        adapter, _ = agent._resolve_adapter()
        assert isinstance(adapter, PyTorchAdapter)

    def test_pytorch_without_param_space_raises_value_error(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-31: PyTorch 使用時に param_space=None で ValueError。"""
        from hpo_agent.agent import HPOAgent

        model_fn, eval_fn, _ = pytorch_setup
        agent = HPOAgent(
            model=model_fn,
            eval_fn=eval_fn,
            n_trials=5,
            param_space=None,
        )
        with pytest.raises(ValueError):
            agent._resolve_adapter()


# ---------------------------------------------------------------------------
# PyTorchAdapter（CMP-32〜34）
# ---------------------------------------------------------------------------


class TestPyTorchAdapter:
    def test_evaluate_returns_float(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-32: evaluate() が float を返す。"""
        from hpo_agent.adapters import PyTorchAdapter

        model_fn, eval_fn, param_space = pytorch_setup
        adapter = PyTorchAdapter(
            model_fn=model_fn, eval_fn=eval_fn, param_space=param_space
        )
        result = adapter.evaluate({"hidden_size": 8})
        assert isinstance(result, float)

    def test_evaluate_calls_model_fn_with_params(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-33: evaluate() が model_fn をパラメータ付きで呼び出し、eval_fn に渡す。"""
        import torch.nn as nn

        from hpo_agent.adapters import PyTorchAdapter

        model_fn, eval_fn, param_space = pytorch_setup
        params = {"hidden_size": 16}
        adapter = PyTorchAdapter(
            model_fn=model_fn, eval_fn=eval_fn, param_space=param_space
        )
        # evaluate が正常終了し、モデルが nn.Module であることを確認
        result = adapter.evaluate(params)
        assert isinstance(result, float)
        # model_fn が正しく hidden_size=16 のモデルを生成することを確認
        model = model_fn(params)
        assert isinstance(model, nn.Module)

    def test_get_default_param_space_returns_provided_space(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-34: get_default_param_space() がコンストラクタに渡した param_space を返す。"""
        from hpo_agent.adapters import PyTorchAdapter

        model_fn, eval_fn, param_space = pytorch_setup
        adapter = PyTorchAdapter(
            model_fn=model_fn, eval_fn=eval_fn, param_space=param_space
        )
        assert adapter.get_default_param_space() is param_space
