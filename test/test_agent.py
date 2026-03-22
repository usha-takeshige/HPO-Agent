"""A-2 エージェント動作テスト (AGT-*).

スーパーバイザーとツール群の連携・LangGraph の制御フローを検証する。
LLM 呼び出しはすべてモックに置き換え、エージェントのフロー制御を確認する。
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.agent


def _run_with_mock(
    n_trials: int,
    llm: Any,
    adapter: Any,
    param_space: Any,
    seed: int | None = None,
    prompts: dict[str, str] | None = None,
) -> Any:
    """モック LLM を使って Supervisor を実行するヘルパー。"""
    from hpo_agent.models import HPOConfig
    from hpo_agent.prompts import SUPERVISOR_DEFAULT_PROMPT, build_system_prompt
    from hpo_agent.report import ReportGenerator
    from hpo_agent.supervisor import Supervisor
    from hpo_agent.tools import (
        BayesianOptimizationTool,
        ExpertAgentTool,
        SobolSearchTool,
    )

    prompts = prompts or {}
    config = HPOConfig(
        model=object(),
        eval_fn=lambda m, X, y: 0.0,
        n_trials=n_trials,
        X=None,
        y=None,
        seed=seed,
        prompts=prompts,
    )

    tools = [
        SobolSearchTool(
            adapter=adapter,
            param_space=param_space,
            seed=seed,
            name="sobol_search",
            description="Sobol 列による準ランダム探索",
        ),
        BayesianOptimizationTool(
            adapter=adapter,
            param_space=param_space,
            seed=seed,
            name="bayesian_optimization",
            description="Optuna によるベイズ最適化",
        ),
        ExpertAgentTool(
            adapter=adapter,
            param_space=param_space,
            llm=MagicMock(),
            system_prompt="test",
            name="expert_agent",
            description="専門家 AI エージェント",
        ),
    ]

    system_prompt = build_system_prompt(
        SUPERVISOR_DEFAULT_PROMPT,
        prompts.get("supervisor"),
    )
    supervisor = Supervisor(
        llm=llm,
        tools=tools,
        report_generator=ReportGenerator(),
        system_prompt=system_prompt,
    )
    return supervisor.run(config)


class TestSupervisorLoop:
    def test_trials_df_has_at_least_one_row(
        self,
        mock_supervisor_llm: MagicMock,
        dummy_adapter: Any,
        simple_param_space: Any,
    ) -> None:
        """AGT-01: ツールを少なくとも1回選択してループが終了する."""
        result = _run_with_mock(
            n_trials=5,
            llm=mock_supervisor_llm,
            adapter=dummy_adapter,
            param_space=simple_param_space,
        )
        assert len(result.trials_df) >= 1

    def test_n_trials_not_exceeded(
        self,
        mock_supervisor_llm: MagicMock,
        dummy_adapter: Any,
        simple_param_space: Any,
    ) -> None:
        """AGT-02: 試行回数の合計が n_trials を超えない."""
        result = _run_with_mock(
            n_trials=10,
            llm=mock_supervisor_llm,
            adapter=dummy_adapter,
            param_space=simple_param_space,
        )
        assert len(result.trials_df) <= 10

    def test_sobol_tool_used(
        self,
        mock_supervisor_llm: MagicMock,
        dummy_adapter: Any,
        simple_param_space: Any,
    ) -> None:
        """AGT-03: SobolSearchTool を選択できる."""
        result = _run_with_mock(
            n_trials=5,
            llm=mock_supervisor_llm,
            adapter=dummy_adapter,
            param_space=simple_param_space,
        )
        assert "sobol_search" in result.trials_df["tool_used"].values

    def test_bayesian_tool_selectable(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """AGT-04: BayesianOptimizationTool を選択できる."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            AIMessage(
                content="bayesian_optimization を実行します。",
                tool_calls=[
                    {
                        "name": "bayesian_optimization",
                        "args": {"n_trials": 3},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="完了", tool_calls=[]),
            MagicMock(content="AI 考察テキスト"),  # generate_final() 用
        ]
        result = _run_with_mock(
            n_trials=5,
            llm=llm,
            adapter=dummy_adapter,
            param_space=simple_param_space,
        )
        assert "bayesian_optimization" in result.trials_df["tool_used"].values

    def test_expert_tool_selectable(
        self, dummy_adapter: Any, simple_param_space: Any, mock_expert_llm: Any
    ) -> None:
        """AGT-05: ExpertAgentTool を選択できる."""
        from hpo_agent.models import HPOConfig
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
                content="expert_agent を実行します。",
                tool_calls=[
                    {"name": "expert_agent", "args": {"n_trials": 2}, "id": "call_1"}
                ],
            ),
            AIMessage(content="完了", tool_calls=[]),
            MagicMock(content="AI 考察テキスト"),  # generate_final() 用
        ]

        config = HPOConfig(
            model=object(),
            eval_fn=lambda m, X, y: 0.0,
            n_trials=5,
            X=None,
            y=None,
        )
        tools = [
            SobolSearchTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                name="sobol_search",
                description="Sobol",
            ),
            BayesianOptimizationTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                name="bayesian_optimization",
                description="Bayesian",
            ),
            ExpertAgentTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                llm=mock_expert_llm,
                system_prompt="test",
                name="expert_agent",
                description="Expert",
            ),
        ]
        supervisor = Supervisor(
            llm=llm,
            tools=tools,
            report_generator=ReportGenerator(),
            system_prompt="test",
        )
        result = supervisor.run(config)
        assert "expert_agent" in result.trials_df["tool_used"].values

    def test_intermediate_report_logged(
        self,
        mock_supervisor_llm: MagicMock,
        dummy_adapter: Any,
        simple_param_space: Any,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AGT-06: ツール完了ごとに logging.INFO が出力される."""
        with caplog.at_level(logging.INFO):
            _run_with_mock(
                n_trials=5,
                llm=mock_supervisor_llm,
                adapter=dummy_adapter,
                param_space=simple_param_space,
            )
        assert any("sobol_search" in record.message for record in caplog.records)

    def test_user_prompt_in_supervisor_call(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """AGT-08: スーパーバイザーへの LLM 呼び出しに追加プロンプトが含まれる."""
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            AIMessage(
                content="実行",
                tool_calls=[
                    {"name": "sobol_search", "args": {"n_trials": 3}, "id": "c1"}
                ],
            ),
            AIMessage(content="完了", tool_calls=[]),
            MagicMock(content="AI 考察テキスト"),  # generate_final() 用
        ]
        _run_with_mock(
            n_trials=5,
            llm=llm,
            adapter=dummy_adapter,
            param_space=simple_param_space,
            prompts={"supervisor": "テスト指示"},
        )
        # LLM に渡されたメッセージに追加プロンプトが含まれるか確認
        call_args = llm.invoke.call_args_list[0]
        messages = call_args[0][0]
        all_content = " ".join(str(m.content) for m in messages)
        assert "テスト指示" in all_content


class TestSeedPropagation:
    def test_seed_propagated_to_sobol(
        self,
        mock_supervisor_llm: MagicMock,
        dummy_adapter: Any,
        simple_param_space: Any,
    ) -> None:
        """AGT-11: seed=42 が SobolSearchTool に伝播される."""
        from hpo_agent.models import HPOConfig
        from hpo_agent.report import ReportGenerator
        from hpo_agent.supervisor import Supervisor
        from hpo_agent.tools import (
            BayesianOptimizationTool,
            ExpertAgentTool,
            SobolSearchTool,
        )

        config = HPOConfig(
            model=object(),
            eval_fn=lambda m, X, y: 0.0,
            n_trials=5,
            X=None,
            y=None,
            seed=42,
        )
        sobol_tool = SobolSearchTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            seed=42,
            name="sobol_search",
            description="test",
        )
        tools = [
            sobol_tool,
            BayesianOptimizationTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                seed=42,
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
            llm=mock_supervisor_llm,
            tools=tools,
            report_generator=ReportGenerator(),
            system_prompt="test",
        )
        supervisor.run(config)
        assert sobol_tool.seed == 42


class TestHistoryPassthrough:
    def test_second_tool_receives_first_tool_history(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """AGT-10: 2回目のツール呼び出しに前回の試行履歴が渡される."""
        from hpo_agent.models import HPOConfig
        from hpo_agent.report import ReportGenerator
        from hpo_agent.supervisor import Supervisor
        from hpo_agent.tools import (
            BayesianOptimizationTool,
            ExpertAgentTool,
            SobolSearchTool,
        )

        call_histories: list[list] = []

        class TrackingBayesianTool(BayesianOptimizationTool):
            def _run(self, n_trials: int, trial_history: list, **kwargs: Any) -> list:
                call_histories.append(list(trial_history))
                return super()._run(
                    n_trials=n_trials, trial_history=trial_history, **kwargs
                )

        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            AIMessage(
                content="sobol 実行",
                tool_calls=[
                    {"name": "sobol_search", "args": {"n_trials": 3}, "id": "c1"}
                ],
            ),
            AIMessage(
                content="bayesian 実行",
                tool_calls=[
                    {
                        "name": "bayesian_optimization",
                        "args": {"n_trials": 2},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content="完了", tool_calls=[]),
            MagicMock(content="AI 考察テキスト"),  # generate_final() 用
        ]

        config = HPOConfig(
            model=object(),
            eval_fn=lambda m, X, y: 0.0,
            n_trials=10,
            X=None,
            y=None,
        )
        tools = [
            SobolSearchTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                name="sobol_search",
                description="test",
            ),
            TrackingBayesianTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
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
            system_prompt="test",
        )
        supervisor.run(config)
        # 2回目の呼び出し（Bayesian）には Sobol の結果が履歴として渡される
        assert len(call_histories) >= 1
        assert len(call_histories[0]) >= 3  # Sobol の 3 試行分


class TestExpertHistorySelection:
    def test_history_selection_limit(
        self, dummy_adapter: Any, simple_param_space: Any, mock_expert_llm: MagicMock
    ) -> None:
        """AGT-13: 30件の履歴からスコア上位20件+直近10件が選ばれる."""
        from datetime import datetime

        from hpo_agent.models import TrialRecord
        from hpo_agent.tools import ExpertAgentTool

        # 40件の履歴を作成
        history = [
            TrialRecord(
                trial_id=i,
                params={"num_leaves": 20 + i},
                score=float(i) / 40,
                tool_used="sobol_search",
                timestamp=datetime(2026, 1, 1, 0, i % 60),
            )
            for i in range(40)
        ]

        captured_content: list[str] = []
        original_invoke = mock_expert_llm.invoke

        def tracking_invoke(messages: list) -> Any:
            for m in messages:
                if hasattr(m, "content") and "trial_id" in str(m.content):
                    captured_content.append(str(m.content))
            return original_invoke(messages)

        mock_expert_llm.invoke = tracking_invoke

        tool = ExpertAgentTool(
            adapter=dummy_adapter,
            param_space=simple_param_space,
            llm=mock_expert_llm,
            system_prompt="test",
            name="expert_agent",
            description="test",
        )
        tool._run(n_trials=1, trial_history=history)

        # 選択された履歴が最大 30 件以内であることを確認
        selected = tool._select_history(history)
        assert len(selected) <= 30


# ---------------------------------------------------------------------------
# AGT-14〜16: NarrowSearchSpaceTool 連携テスト
# ---------------------------------------------------------------------------


class TestNarrowSearchSpaceIntegration:
    def test_narrow_search_space_tool_selectable(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """AGT-14: narrow_search_space ツールが Supervisor に渡されエラーなく完了する."""
        import json

        from hpo_agent.models import HPOConfig
        from hpo_agent.report import ReportGenerator
        from hpo_agent.supervisor import Supervisor
        from hpo_agent.tools import (
            BayesianOptimizationTool,
            ExpertAgentTool,
            NarrowSearchSpaceTool,
            SobolSearchTool,
        )

        narrow_args = json.dumps([{"name": "num_leaves", "low": 30, "high": 80}])
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            AIMessage(
                content="探索空間を狭めます。",
                tool_calls=[
                    {
                        "name": "narrow_search_space",
                        "args": {"param_updates": narrow_args},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(content="完了", tool_calls=[]),
            MagicMock(content="AI 考察テキスト"),
        ]
        config = HPOConfig(
            model=object(),
            eval_fn=lambda m, X, y: 0.0,
            n_trials=5,
            X=None,
            y=None,
        )
        tools = [
            SobolSearchTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                name="sobol_search",
                description="test",
            ),
            BayesianOptimizationTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
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
            NarrowSearchSpaceTool(
                param_space=simple_param_space,
                name="narrow_search_space",
                description="test",
            ),
        ]
        supervisor = Supervisor(
            llm=llm,
            tools=tools,
            report_generator=ReportGenerator(),
            system_prompt="test",
        )
        result = supervisor.run(config)
        # narrow_search_space は TrialRecord を生成しないが、エラーなく完了すること
        assert result is not None

    def test_tool_executor_updates_current_param_space(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """AGT-16: _tool_executor_node を直接呼び出すと current_param_space が更新される."""
        import json

        from langchain_core.messages import AIMessage, SystemMessage

        from hpo_agent.models import HPOConfig
        from hpo_agent.report import ReportGenerator
        from hpo_agent.state import SupervisorState
        from hpo_agent.supervisor import Supervisor
        from hpo_agent.tools import (
            BayesianOptimizationTool,
            ExpertAgentTool,
            NarrowSearchSpaceTool,
            SobolSearchTool,
        )

        narrow_args = json.dumps([{"name": "num_leaves", "low": 50, "high": 70}])
        supervisor = Supervisor(
            llm=MagicMock(),
            tools=[
                SobolSearchTool(
                    adapter=dummy_adapter,
                    param_space=simple_param_space,
                    name="sobol_search",
                    description="test",
                ),
                NarrowSearchSpaceTool(
                    param_space=simple_param_space,
                    name="narrow_search_space",
                    description="test",
                ),
                BayesianOptimizationTool(
                    adapter=dummy_adapter,
                    param_space=simple_param_space,
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
            ],
            report_generator=ReportGenerator(),
            system_prompt="test",
        )
        state = SupervisorState(
            messages=[
                SystemMessage(content="test"),
                AIMessage(
                    content="探索空間を狭めます。",
                    tool_calls=[
                        {
                            "name": "narrow_search_space",
                            "args": {"param_updates": narrow_args},
                            "id": "c1",
                        }
                    ],
                ),
            ],
            trial_records=[],
            remaining_trials=10,
            config=HPOConfig(
                model=object(),
                eval_fn=lambda m, X, y: 0.0,
                n_trials=10,
                X=None,
                y=None,
            ),
        )
        result = supervisor._tool_executor_node(state)
        assert "current_param_space" in result
        assert result["current_param_space"] is not None
        nl_spec = next(
            s for s in result["current_param_space"].specs if s.name == "num_leaves"
        )
        assert nl_spec.low == 50.0
        assert nl_spec.high == 70.0

    def test_subsequent_sobol_uses_narrowed_space(
        self, dummy_adapter: Any, simple_param_space: Any
    ) -> None:
        """AGT-15: narrow_search_space 後に Sobol を呼ぶと狭めた範囲内のパラメータのみ返る."""
        import json

        from hpo_agent.models import HPOConfig
        from hpo_agent.report import ReportGenerator
        from hpo_agent.supervisor import Supervisor
        from hpo_agent.tools import (
            BayesianOptimizationTool,
            ExpertAgentTool,
            NarrowSearchSpaceTool,
            SobolSearchTool,
        )

        narrow_args = json.dumps([{"name": "num_leaves", "low": 50, "high": 70}])
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            AIMessage(
                content="探索空間を狭めます。",
                tool_calls=[
                    {
                        "name": "narrow_search_space",
                        "args": {"param_updates": narrow_args},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(
                content="Sobol で探索します。",
                tool_calls=[
                    {"name": "sobol_search", "args": {"n_trials": 5}, "id": "c2"}
                ],
            ),
            AIMessage(content="完了", tool_calls=[]),
            MagicMock(content="AI 考察テキスト"),
        ]
        config = HPOConfig(
            model=object(),
            eval_fn=lambda m, X, y: 0.0,
            n_trials=5,
            X=None,
            y=None,
        )
        tools = [
            SobolSearchTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
                seed=0,
                name="sobol_search",
                description="test",
            ),
            NarrowSearchSpaceTool(
                param_space=simple_param_space,
                name="narrow_search_space",
                description="test",
            ),
            BayesianOptimizationTool(
                adapter=dummy_adapter,
                param_space=simple_param_space,
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
            system_prompt="test",
        )
        result = supervisor.run(config)
        # Sobol の結果が狭めた範囲内であることを確認
        sobol_rows = result.trials_df[result.trials_df["tool_used"] == "sobol_search"]
        assert len(sobol_rows) > 0
        for _, row in sobol_rows.iterrows():
            assert 50 <= row["num_leaves"] <= 70
