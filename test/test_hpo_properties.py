"""A-3 HPO 性質テスト (HPO-*).

HPO として期待される最適化の性質が成立するかを検証する。
軽量なダミーモデルと確定的な eval 関数を使い、実際にエージェントを動作させる。
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.hpo


def _make_sobol_result(
    dummy_adapter: Any,
    simple_param_space: Any,
    n_trials: int = 10,
    seed: int = 42,
) -> list:
    """SobolSearchTool を直接実行して TrialRecord リストを返すヘルパー。"""
    from hpo_agent.tools import SobolSearchTool

    tool = SobolSearchTool(
        adapter=dummy_adapter,
        param_space=simple_param_space,
        seed=seed,
        name="sobol_search",
        description="test",
    )
    return tool._run(n_trials=n_trials, trial_history=[])


def _run_supervisor_with_sobol(
    dummy_adapter: Any,
    simple_param_space: Any,
    n_trials: int = 10,
    seed: int = 42,
) -> Any:
    """MockSupervisorLLM (sobol_search のみ) で Supervisor を実行するヘルパー。"""
    from hpo_agent.models import HPOConfig
    from hpo_agent.prompts import SUPERVISOR_DEFAULT_PROMPT
    from hpo_agent.report import ReportGenerator
    from hpo_agent.supervisor import Supervisor
    from hpo_agent.tools import (
        BayesianOptimizationTool,
        ExpertAgentTool,
        SobolSearchTool,
    )

    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.invoke.side_effect = [
        AIMessage(
            content="sobol_search を実行します。",
            tool_calls=[
                {"name": "sobol_search", "args": {"n_trials": n_trials}, "id": "c1"}
            ],
        ),
        AIMessage(content="完了", tool_calls=[]),
        MagicMock(content="AI 考察テキスト"),  # generate_final() 用
    ]

    config = HPOConfig(
        model=object(),
        eval_fn=lambda m, X, y: 0.0,
        n_trials=n_trials,
        X=None,
        y=None,
        seed=seed,
    )
    tools = [
        SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=seed,
            name="sobol_search",
            description="test",
        ),
        BayesianOptimizationTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=seed,
            name="bayesian_optimization",
            description="test",
        ),
        ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=MagicMock(),
            system_prompt="test",
            name="expert_agent",
            description="test",
        ),
    ]
    supervisor = Supervisor(
        llm=llm,
        tools=tools,
        report_generator=ReportGenerator(),
        system_prompt=SUPERVISOR_DEFAULT_PROMPT,
    )
    return supervisor.run(config)


class TestResultCorrectness:
    def test_best_score_equals_max(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-01: best_score が全試行の最良値と一致する."""
        result = _run_supervisor_with_sobol(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=5,
        )
        assert result.best_score == result.trials_df["score"].max()

    def test_trials_df_row_count_matches_n_trials(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-10: trials_df の行数が n_trials 以内."""
        n_trials = 5
        result = _run_supervisor_with_sobol(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=n_trials,
        )
        assert len(result.trials_df) <= n_trials

    def test_best_params_achieves_best_score(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-02: best_params を eval_fn に渡すと best_score と一致する."""
        result = _run_supervisor_with_sobol(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=5,
        )
        # DummyAdapter は常に 0.85 を返すため、best_params の評価結果も 0.85 になる
        score = dummy_adapter.evaluate(result.best_params)
        assert abs(score - result.best_score) < 1e-6


class TestSeedReproducibility:
    def test_sobol_reproducible_with_seed(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-04: Sobol 探索が同一シードで再現可能."""
        results_a = _make_sobol_result(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=10,
            seed=42,
        )
        results_b = _make_sobol_result(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=10,
            seed=42,
        )
        params_a = [r.params for r in results_a]
        params_b = [r.params for r in results_b]
        assert params_a == params_b

    def test_bayesian_reproducible_with_seed(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-03: ベイズ最適化が同一シードで再現可能."""
        from hpo_agent.tools import BayesianOptimizationTool

        tool_a = BayesianOptimizationTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=42,
            name="bayesian_optimization",
            description="test",
        )
        tool_b = BayesianOptimizationTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=42,
            name="bayesian_optimization",
            description="test",
        )
        results_a = tool_a._run(n_trials=5, trial_history=[])
        results_b = tool_b._run(n_trials=5, trial_history=[])
        assert [r.params for r in results_a] == [r.params for r in results_b]


class TestSobolCoverage:
    def test_sobol_covers_param_space_uniformly(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-05: Sobol 20試行で num_leaves の全4分位をカバー."""
        results = _make_sobol_result(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=20,
            seed=0,
        )
        values = [r.params["num_leaves"] for r in results]
        # [20, 40), [40, 60), [60, 80), [80, 100] の4区間
        bins = [20, 40, 60, 80, 100]
        for lo, hi in zip(bins[:-1], bins[1:]):
            assert any(
                lo <= v <= hi for v in values
            ), f"区間 [{lo}, {hi}] に探索点がない. values={values}"


class TestParamSpaceConstraints:
    def test_all_params_within_bounds(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-07: 全試行のパラメータが param_space 制約を満たす."""
        results = _make_sobol_result(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=20,
            seed=42,
        )
        for r in results:
            assert 20 <= r.params["num_leaves"] <= 100
            assert 0.01 <= r.params["learning_rate"] <= 0.3
            assert r.params["boosting_type"] in ("gbdt", "dart")


class TestTimingMeasurement:
    def test_eval_duration_positive(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-08: 全試行の eval_duration が正の値（DummyAdapter は sleep(0.001) あり）."""
        results = _make_sobol_result(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=5,
            seed=42,
        )
        assert all(r.eval_duration > 0 for r in results)

    def test_algo_duration_non_negative(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-09: 全試行の algo_duration が非負."""
        results = _make_sobol_result(
            dummy_adapter=dummy_adapter,
            simple_param_space=simple_param_space,
            n_trials=5,
            seed=42,
        )
        assert all(r.algo_duration >= 0 for r in results)


class TestStability:
    def test_run_three_times_without_crash(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-10: 同一条件で3回実行してもクラッシュしない（安定性）."""
        for _ in range(3):
            _run_supervisor_with_sobol(
                dummy_adapter=dummy_adapter,
                simple_param_space=simple_param_space,
                n_trials=3,
                seed=0,
            )


class TestNarrowedSpaceConstraints:
    def test_params_within_narrowed_bounds(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """HPO-07b: SobolSearchTool に narrow した ParamSpace を渡すと全試行が狭めた bounds 内."""
        from hpo_agent.models import ParamSpace, ParamSpec
        from hpo_agent.tools import SobolSearchTool

        narrow_space = ParamSpace(
            specs=(
                ParamSpec(name="num_leaves", type="int", low=40, high=70),
                ParamSpec(
                    name="learning_rate", type="float", low=0.05, high=0.2, log=True
                ),
                ParamSpec(name="boosting_type", type="categorical", choices=("gbdt",)),
            )
        )
        tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=42,
            name="sobol_search",
            description="test",
        )
        results = tool._run(
            n_trials=20,
            trial_history=[],
            effective_param_space=narrow_space,
        )
        for r in results:
            assert 40 <= r.params["num_leaves"] <= 70
            assert 0.05 <= r.params["learning_rate"] <= 0.2
            assert r.params["boosting_type"] == "gbdt"
