"""A-1 コンポーネント実装テスト (CMP-*).

各クラス・メソッドが仕様通りに動作するかを検証する。
LLM 呼び出しはすべてモックに置き換える。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from hpo_agent.models import ParamSpace

if TYPE_CHECKING:
    from conftest import DummyAdapter

pytestmark = pytest.mark.component


# ---------------------------------------------------------------------------
# OpenAILLMProvider / _resolve_llm_provider
# ---------------------------------------------------------------------------


class TestOpenAILLMProvider:
    def test_get_llm_returns_base_chat_model(self) -> None:
        """OpenAILLMProvider.get_llm() が BaseChatModel を返す。"""
        from unittest.mock import MagicMock, patch

        from langchain_core.language_models import BaseChatModel

        from hpo_agent.providers import OpenAILLMProvider

        mock_llm = MagicMock(spec=BaseChatModel)
        with patch("langchain_openai.ChatOpenAI", return_value=mock_llm):
            provider = OpenAILLMProvider(api_key="sk-dummy", model_name="gpt-4o")
            llm = provider.get_llm(temperature=0)
        assert llm is mock_llm

    def test_resolve_llm_provider_returns_openai(
        self, lgbm_binary_setup, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        """LLM_PROVIDER=openai で OpenAILLMProvider が返る。"""
        from hpo_agent.agent import HPOAgent
        from hpo_agent.providers import OpenAILLMProvider

        model, eval_fn, X, y = lgbm_binary_setup
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_API_KEY", "sk-dummy")
        monkeypatch.setenv("LLM_MODEL_NAME", "gpt-4o")

        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        provider = agent._resolve_llm_provider()
        assert isinstance(provider, OpenAILLMProvider)

    def test_resolve_llm_provider_unknown_raises(
        self, lgbm_binary_setup, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        """未サポートの LLM_PROVIDER 値で ValueError が送出される。"""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = lgbm_binary_setup
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        monkeypatch.setenv("LLM_MODEL_NAME", "dummy-model")

        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            agent._resolve_llm_provider()


# ---------------------------------------------------------------------------
# AnthropicLLMProvider / _resolve_llm_provider
# ---------------------------------------------------------------------------


class TestAnthropicLLMProvider:
    def test_get_llm_returns_base_chat_model(self) -> None:
        """AnthropicLLMProvider.get_llm() が BaseChatModel を返す。"""
        from unittest.mock import MagicMock, patch

        from langchain_core.language_models import BaseChatModel

        from hpo_agent.providers import AnthropicLLMProvider

        mock_llm = MagicMock(spec=BaseChatModel)
        with patch("langchain_anthropic.ChatAnthropic", return_value=mock_llm):
            provider = AnthropicLLMProvider(
                api_key="sk-ant-dummy", model_name="claude-opus-4-6"
            )
            llm = provider.get_llm(temperature=0)
        assert llm is mock_llm

    def test_resolve_llm_provider_returns_anthropic(
        self, lgbm_binary_setup, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        """LLM_PROVIDER=anthropic で AnthropicLLMProvider が返る。"""
        from hpo_agent.agent import HPOAgent
        from hpo_agent.providers import AnthropicLLMProvider

        model, eval_fn, X, y = lgbm_binary_setup
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("LLM_API_KEY", "sk-ant-dummy")
        monkeypatch.setenv("LLM_MODEL_NAME", "claude-opus-4-6")

        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        provider = agent._resolve_llm_provider()
        assert isinstance(provider, AnthropicLLMProvider)

    def test_resolve_llm_provider_unknown_raises(
        self, lgbm_binary_setup, monkeypatch: pytest.MonkeyPatch  # type: ignore[no-untyped-def]
    ) -> None:
        """未サポートの LLM_PROVIDER 値で ValueError が送出される。"""
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = lgbm_binary_setup
        monkeypatch.setenv("LLM_PROVIDER", "unknown_provider")
        monkeypatch.setenv("LLM_API_KEY", "dummy")
        monkeypatch.setenv("LLM_MODEL_NAME", "dummy-model")

        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            agent._resolve_llm_provider()


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

    def test_is_partial_true_when_low_missing(self) -> None:
        """is_partial: low が None の数値型は部分指定."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="lr", type="float", high=1.0)
        assert spec.is_partial is True

    def test_is_partial_true_when_high_missing(self) -> None:
        """is_partial: high が None の数値型は部分指定."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="n_layers", type="int", low=1)
        assert spec.is_partial is True

    def test_is_partial_true_when_both_bounds_missing(self) -> None:
        """is_partial: low/high ともに None の数値型は部分指定."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="n_layers", type="int")
        assert spec.is_partial is True

    def test_is_partial_true_when_choices_missing_categorical(self) -> None:
        """is_partial: choices が None の categorical 型は部分指定."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="activation", type="categorical")
        assert spec.is_partial is True

    def test_is_partial_false_when_fully_specified_numeric(self) -> None:
        """is_partial: low/high が揃っている数値型は完全指定."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="lr", type="float", low=0.001, high=1.0)
        assert spec.is_partial is False

    def test_is_partial_false_when_fully_specified_categorical(self) -> None:
        """is_partial: choices が設定されている categorical 型は完全指定."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(
            name="activation", type="categorical", choices=("relu", "tanh")
        )
        assert spec.is_partial is False

    def test_log_true_partial_does_not_raise(self) -> None:
        """log=True かつ部分指定（low=None）の場合はバリデーションエラーが発生しない."""
        from hpo_agent.models import ParamSpec

        spec = ParamSpec(name="lr", type="float", log=True)
        assert spec.is_partial is True


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

    def test_get_default_param_space_raises(self, lgbm_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-24b: get_default_param_space() は NotImplementedError を送出する（LLM が自動生成）。"""
        from hpo_agent.adapters import LightGBMAdapter

        model, eval_fn, X, y = lgbm_binary_setup
        adapter = LightGBMAdapter(model=model, eval_fn=eval_fn, X=X, y=y)
        with pytest.raises(NotImplementedError):
            adapter.get_default_param_space()


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

    def test_generate_intermediate_with_space_change(
        self,
        sample_trial_records: list,
        search_space_change_record: Any,
    ) -> None:
        """CMP-SP-01: latest_space_change 指定時に探索空間の変更通知セクションが含まれる."""
        from hpo_agent.report import ReportGenerator

        gen = ReportGenerator()
        report = gen.generate_intermediate(
            trial_records=sample_trial_records,
            best_params={"num_leaves": 60},
            best_score=0.90,
            latest_space_change=search_space_change_record,
        )
        assert "探索空間の変更通知" in report
        assert "変更前" in report
        assert "変更後" in report

    def test_generate_intermediate_without_space_change(
        self,
        sample_trial_records: list,
    ) -> None:
        """CMP-SP-02: latest_space_change=None のとき変更通知セクションが含まれない."""
        from hpo_agent.report import ReportGenerator

        gen = ReportGenerator()
        report = gen.generate_intermediate(
            trial_records=sample_trial_records,
            best_params={"num_leaves": 60},
            best_score=0.90,
            latest_space_change=None,
        )
        assert "探索空間の変更通知" not in report

    def test_generate_final_with_space_change_history(
        self,
        sample_trial_records: list,
        search_space_change_record: Any,
    ) -> None:
        """CMP-SP-03: search_space_change_history 非空のとき変更履歴セクションが含まれる."""
        from unittest.mock import MagicMock

        from hpo_agent.report import ReportGenerator

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="AI 考察テスト")

        gen = ReportGenerator()
        report = gen.generate_final(
            trial_records=sample_trial_records,
            best_params={"num_leaves": 60},
            best_score=0.90,
            llm=mock_llm,
            search_space_change_history=[search_space_change_record],
        )
        assert "探索空間の変更履歴" in report
        assert "変更前" in report
        assert "変更後" in report
        assert "試行 5 件完了後" in report

    def test_generate_final_without_space_change_history(
        self,
        sample_trial_records: list,
    ) -> None:
        """CMP-SP-04: search_space_change_history が空のとき変更履歴セクションが含まれない."""
        from unittest.mock import MagicMock

        from hpo_agent.report import ReportGenerator

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="AI 考察テスト")

        gen = ReportGenerator()
        report = gen.generate_final(
            trial_records=sample_trial_records,
            best_params={"num_leaves": 60},
            best_score=0.90,
            llm=mock_llm,
            search_space_change_history=[],
        )
        assert "探索空間の変更履歴" not in report


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

    def test_resolve_adapter_without_param_space_returns_none(self, sklearn_binary_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-29: sklearn モデルで param_space を省略すると (adapter, None) が返る（ValueError なし）。"""
        from hpo_agent.adapters import SklearnAdapter
        from hpo_agent.agent import HPOAgent

        model, eval_fn, X, y = sklearn_binary_setup
        agent = HPOAgent(model=model, eval_fn=eval_fn, n_trials=5, X=X, y=y)
        adapter, param_space = agent._resolve_adapter()
        assert isinstance(adapter, SklearnAdapter)
        assert param_space is None

    def test_pytorch_model_fn_resolves_to_pytorch_adapter(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-30: callable な model_fn が PyTorchAdapter に解決される（param_space 不要）。"""
        from hpo_agent.adapters import PyTorchAdapter
        from hpo_agent.agent import HPOAgent

        model_fn, eval_fn, _ = pytorch_setup
        agent = HPOAgent(model=model_fn, eval_fn=eval_fn, n_trials=5)
        adapter, _ = agent._resolve_adapter()
        assert isinstance(adapter, PyTorchAdapter)

    def test_pytorch_without_param_space_returns_none(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-31: PyTorch 使用時に param_space=None で (adapter, None) が返る（ValueError なし）。"""
        from hpo_agent.adapters import PyTorchAdapter
        from hpo_agent.agent import HPOAgent

        model_fn, eval_fn, _ = pytorch_setup
        agent = HPOAgent(model=model_fn, eval_fn=eval_fn, n_trials=5, param_space=None)
        adapter, param_space = agent._resolve_adapter()
        assert isinstance(adapter, PyTorchAdapter)
        assert param_space is None


# ---------------------------------------------------------------------------
# PyTorchAdapter（CMP-32〜34）
# ---------------------------------------------------------------------------


class TestPyTorchAdapter:
    def test_evaluate_returns_float(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-32: evaluate() が float を返す。"""
        from hpo_agent.adapters import PyTorchAdapter

        model_fn, eval_fn, _ = pytorch_setup
        adapter = PyTorchAdapter(model_fn=model_fn, eval_fn=eval_fn)
        result = adapter.evaluate({"hidden_size": 8})
        assert isinstance(result, float)

    def test_evaluate_calls_model_fn_with_params(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-33: evaluate() が model_fn をパラメータ付きで呼び出し、eval_fn に渡す。"""
        import torch.nn as nn

        from hpo_agent.adapters import PyTorchAdapter

        model_fn, eval_fn, _ = pytorch_setup
        params = {"hidden_size": 16}
        adapter = PyTorchAdapter(model_fn=model_fn, eval_fn=eval_fn)
        result = adapter.evaluate(params)
        assert isinstance(result, float)
        model = model_fn(params)
        assert isinstance(model, nn.Module)

    def test_get_default_param_space_raises(self, pytorch_setup) -> None:  # type: ignore[no-untyped-def]
        """CMP-34: get_default_param_space() は NotImplementedError を送出する（LLM が自動生成）。"""
        from hpo_agent.adapters import PyTorchAdapter

        model_fn, eval_fn, _ = pytorch_setup
        adapter = PyTorchAdapter(model_fn=model_fn, eval_fn=eval_fn)
        with pytest.raises(NotImplementedError):
            adapter.get_default_param_space()


# ---------------------------------------------------------------------------
# CMP-44/45: ParamSpecSchema / ParamSpaceSchema
# ---------------------------------------------------------------------------


class TestParamSpaceSchema:
    def test_to_param_space_returns_frozen_dataclass(self) -> None:
        """CMP-44: ParamSpaceSchema.to_param_space() が frozen ParamSpace を返す。"""
        from dataclasses import FrozenInstanceError

        from hpo_agent.models import ParamSpaceSchema

        schema = ParamSpaceSchema.model_validate(
            {
                "specs": [
                    {"name": "lr", "type": "float", "low": 0.001, "high": 0.3},
                    {"name": "depth", "type": "int", "low": 2, "high": 10},
                ]
            }
        )
        space = schema.to_param_space()
        assert isinstance(space, ParamSpace)
        assert len(space.specs) == 2
        with pytest.raises(FrozenInstanceError):
            space.specs = ()  # type: ignore[misc]

    def test_categorical_choices_list_to_tuple(self) -> None:
        """CMP-45: ParamSpecSchema の choices=list が to_param_spec() で tuple に変換される。"""
        from hpo_agent.models import ParamSpecSchema

        schema = ParamSpecSchema.model_validate(
            {"name": "bt", "type": "categorical", "choices": ["gbdt", "dart"]}
        )
        spec = schema.to_param_spec()
        assert isinstance(spec.choices, tuple)
        assert spec.choices == ("gbdt", "dart")


# ---------------------------------------------------------------------------
# CMP-35〜42: ChangeSearchSpaceTool
# ---------------------------------------------------------------------------


class TestChangeSearchSpaceTool:
    def test_valid_update_returns_description(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-35: 有効な param_updates で説明文字列を返す（"Error" を含まない）."""
        import json

        updates = json.dumps([{"name": "num_leaves", "low": 30, "high": 80}])
        result = change_search_space_tool._run(param_updates=updates)
        assert isinstance(result, str)
        assert "Error" not in result
        assert "num_leaves" in result

    def test_unknown_param_name_returns_error(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-36: 存在しないパラメータ名 → "Error" を含む文字列を返す."""
        import json

        updates = json.dumps([{"name": "nonexistent", "low": 0.1, "high": 0.2}])
        result = change_search_space_tool._run(param_updates=updates)
        assert "Error" in result

    def test_expanding_beyond_original_succeeds(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-37: 数値型の low を元より小さく（拡大方向に）指定しても成功する."""
        import json

        # num_leaves の original low は 20 → 10 に拡大しても OK
        updates = json.dumps([{"name": "num_leaves", "low": 10, "high": 80}])
        result = change_search_space_tool._run(param_updates=updates)
        assert "Error" not in result
        assert "num_leaves" in result

    def test_expanding_categorical_with_new_choices_succeeds(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-38: categorical で元にない選択肢を追加しても成功する."""
        import json

        # boosting_type の original choices は ("gbdt", "dart") → 新たに "goss" を追加
        updates = json.dumps([{"name": "boosting_type", "choices": ["gbdt", "goss"]}])
        result = change_search_space_tool._run(param_updates=updates)
        assert "Error" not in result
        assert "boosting_type" in result

    def test_empty_categorical_choices_returns_error(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-38b: categorical で空の choices を指定 → "Error" を含む文字列."""
        import json

        updates = json.dumps([{"name": "boosting_type", "choices": []}])
        result = change_search_space_tool._run(param_updates=updates)
        assert "Error" in result

    def test_low_greater_than_high_returns_error(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-38c: new_low >= new_high の場合 → "Error" を含む文字列."""
        import json

        updates = json.dumps([{"name": "num_leaves", "low": 80, "high": 30}])
        result = change_search_space_tool._run(param_updates=updates)
        assert "Error" in result

    def test_build_changed_space_returns_param_space(
        self, change_search_space_tool: Any
    ) -> None:
        """CMP-39: _build_changed_space が有効入力で ParamSpace を返し、値が反映される."""
        import json

        from hpo_agent.models import ParamSpace

        updates = json.dumps([{"name": "num_leaves", "low": 30, "high": 80}])
        result = change_search_space_tool._build_changed_space(updates)
        assert isinstance(result, ParamSpace)
        nl_spec = next(s for s in result.specs if s.name == "num_leaves")
        assert nl_spec.low == 30.0
        assert nl_spec.high == 80.0

    def test_unmodified_params_preserved(self, change_search_space_tool: Any) -> None:
        """CMP-40: 更新対象でないパラメータは元の値を引き継ぐ."""
        import json

        from hpo_agent.models import ParamSpace

        updates = json.dumps([{"name": "num_leaves", "low": 30, "high": 80}])
        result = change_search_space_tool._build_changed_space(updates)
        assert isinstance(result, ParamSpace)
        lr_spec = next(s for s in result.specs if s.name == "learning_rate")
        assert lr_spec.low == 0.01
        assert lr_spec.high == 0.3

    def test_sobol_uses_effective_param_space(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """CMP-41: SobolSearchTool に effective_param_space を渡すと指定した範囲内でサンプリング."""
        from hpo_agent.models import ParamSpace, ParamSpec
        from hpo_agent.tools import SobolSearchTool

        changed_space = ParamSpace(
            specs=(
                ParamSpec(name="num_leaves", type="int", low=50, high=70),
                ParamSpec(
                    name="learning_rate", type="float", low=0.05, high=0.1, log=True
                ),
                ParamSpec(name="boosting_type", type="categorical", choices=("gbdt",)),
            )
        )
        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="sobol_search",
            description="test",
        )
        results = tool._run(
            n_trials=10, trial_history=[], effective_param_space=changed_space
        )
        for r in results:
            assert 50 <= r.params["num_leaves"] <= 70
            assert 0.05 <= r.params["learning_rate"] <= 0.1
            assert r.params["boosting_type"] == "gbdt"

    def test_sobol_uses_expanded_param_space(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """CMP-42: 拡大した空間を渡すと元の上限を超える範囲でもサンプリングされる."""
        from hpo_agent.models import ParamSpace, ParamSpec
        from hpo_agent.tools import SobolSearchTool

        # num_leaves を original の high=200 より大きい 500 まで拡大
        expanded_space = ParamSpace(
            specs=(
                ParamSpec(name="num_leaves", type="int", low=20, high=500),
                ParamSpec(
                    name="learning_rate", type="float", low=0.01, high=0.3, log=True
                ),
                ParamSpec(
                    name="boosting_type",
                    type="categorical",
                    choices=("gbdt", "dart"),
                ),
            )
        )
        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=0,
            name="sobol_search",
            description="test",
        )
        results = tool._run(
            n_trials=20, trial_history=[], effective_param_space=expanded_space
        )
        # すべての試行が拡大後の範囲内に収まっていることを確認
        for r in results:
            assert 20 <= r.params["num_leaves"] <= 500
