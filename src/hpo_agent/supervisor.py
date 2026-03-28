"""LangGraph-based supervisor orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Literal

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from hpo_agent.models import HPOConfig, HPOResult, ParamSpace, TrialRecord
from hpo_agent.report import ReportGenerator
from hpo_agent.state import SupervisorState
from hpo_agent.tools import ChangeSearchSpaceTool, HPOToolBase, _describe_param_space

logger = logging.getLogger(__name__)


class Supervisor:
    """LangGraph グラフを構築し、ツール選択・実行ループを制御して HPOResult を返す。

    依存関係はコンストラクタで注入（DI）する。
    ツールの追加は tools リストへの追加のみで対応可能（OCP 準拠）。

    Args:
        llm: Supervisor が使用する LLM インスタンス。
        tools: 使用可能なツールのリスト（HPOToolBase および ChangeSearchSpaceTool）。
        report_generator: 中間・最終レポート生成器。
        system_prompt: Supervisor への初期システムプロンプト。
        generated_param_space: LLM が自動生成したパラメータ空間。レポートに記載される。
    """

    def __init__(
        self,
        llm: Any,
        tools: list[BaseTool],
        report_generator: ReportGenerator,
        system_prompt: str,
        generated_param_space: ParamSpace | None = None,
    ) -> None:
        """Supervisor を初期化する。"""
        self._llm = llm
        self._tools = tools
        self._report_generator = report_generator
        self._system_prompt = system_prompt
        self._generated_param_space = generated_param_space
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in tools}

    def run(self, config: HPOConfig) -> HPOResult:

        """LangGraph グラフを実行して HPOResult を返す。

        Args:
            config: エージェント実行設定。

        Returns:
            最適化結果（HPOResult）。
        """
        graph = self._build_graph()
        initial_state = SupervisorState(
            messages=[
                SystemMessage(content=self._system_prompt),
                HumanMessage(
                    content=(
                        f"ハイパーパラメーター最適化を開始してください。"
                        f"総試行回数: {config.n_trials} 回。"
                        f"利用可能なツールを選択して最適化を進めてください。"
                    )
                ),
            ],
            trial_records=[],
            remaining_trials=config.n_trials,
            config=config,
        )
        final_state = graph.invoke(initial_state)
        return self._build_result(final_state, config)

    def _build_graph(self) -> Any:
        """LangGraph StateGraph を構築してコンパイルする。"""
        workflow = StateGraph(SupervisorState)
        workflow.add_node("supervisor_node", self._supervisor_node)
        workflow.add_node("tool_executor_node", self._tool_executor_node)

        workflow.set_entry_point("supervisor_node")
        workflow.add_conditional_edges(
            "supervisor_node",
            self._should_continue,
            {"continue": "tool_executor_node", "end": END},
        )
        workflow.add_edge("tool_executor_node", "supervisor_node")

        return workflow.compile()

    def _supervisor_node(self, state: SupervisorState) -> dict[str, Any]:
        """LLM を呼び出してツール選択を決定するノード。"""
        llm_with_tools = self._llm.bind_tools(self._tools)
        response: AIMessage = llm_with_tools.invoke(state.messages)
        _content = response.content
        if isinstance(_content, list):
            _content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in _content
            )
        reasoning = _content if isinstance(_content, str) else ""
        if response.tool_calls:
            _tc = response.tool_calls[0]
            logger.info(
                "Supervisor selected tool '%s' with %d planned trial(s).",
                _tc["name"],
                int(_tc.get("args", {}).get("n_trials", 1)),
            )
        return {
            "messages": [response],
            "last_tool_reasoning": reasoning,
        }

    def _tool_executor_node(self, state: SupervisorState) -> dict[str, Any]:
        """選択されたツールを実行して試行結果を状態に追加するノード。"""
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        tool_call = last_message.tool_calls[0]
        tool_name: str = tool_call["name"]
        args: dict[str, Any] = tool_call["args"]
        tool_call_id: str = tool_call.get("id") or ""

        tool = self._tool_map.get(tool_name)
        if tool is None:
            logger.warning("Unknown tool: %s", tool_name)
            return {
                "messages": [
                    ToolMessage(
                        content=f"Unknown tool: {tool_name}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }

        # ChangeSearchSpaceTool は試行を実行しない専用処理
        if isinstance(tool, ChangeSearchSpaceTool):
            return self._handle_change_search_space(tool, args, tool_call_id)

        assert isinstance(tool, HPOToolBase)
        requested_trials = int(args.get("n_trials", 1))
        n_trials = min(requested_trials, state.remaining_trials)
        logger.info(
            "Executing tool '%s': %d trial(s) (requested: %d, remaining: %d).",
            tool_name,
            n_trials,
            requested_trials,
            state.remaining_trials,
        )

        new_records = tool._run(
            n_trials=n_trials,
            trial_history=list(state.trial_records),
            effective_param_space=state.current_param_space,
        )

        # trial_id を既存件数からオフセット
        start_id = len(state.trial_records)
        for i, record in enumerate(new_records):
            record.trial_id = start_id + i
            # Supervisor のツール選択理由を reasoning として補完（ExpertAgentTool 以外）
            if not record.reasoning and state.last_tool_reasoning:
                record.reasoning = state.last_tool_reasoning

        updated_records = list(state.trial_records) + new_records
        remaining = state.remaining_trials - len(new_records)

        # 中間レポートを生成してログ出力
        best_score, best_params = self._find_best(updated_records)
        report = self._report_generator.generate_intermediate(
            trial_records=updated_records,
            best_params=best_params,
            best_score=best_score,
            seed=state.config.seed,
            tool_reasoning=state.last_tool_reasoning,
            current_tool_records=new_records,
        )
        logger.info(
            "Tool '%s' completed (%d trials). Best score: %.6f\n%s",
            tool_name,
            len(new_records),
            best_score,
            report,
        )

        tool_message = ToolMessage(
            content=(
                f"{tool_name}: {len(new_records)} trials completed. "
                f"Best score: {best_score:.6f}\n\n{report}"
            ),
            tool_call_id=tool_call_id,
        )

        return {
            "messages": [tool_message],
            "trial_records": updated_records,
            "remaining_trials": remaining,
            "current_report": report,
        }

    def _handle_change_search_space(
        self,
        tool: ChangeSearchSpaceTool,
        args: dict[str, Any],
        tool_call_id: str,
    ) -> dict[str, Any]:
        """ChangeSearchSpaceTool の実行結果を処理して current_param_space を更新する。

        Args:
            tool: ChangeSearchSpaceTool インスタンス。
            args: ツール呼び出し引数。
            tool_call_id: ツール呼び出し ID。

        Returns:
            状態更新辞書。成功時は current_param_space を含む。
        """
        param_updates: str = args.get("param_updates", "[]")
        result = tool._build_changed_space(param_updates)
        if isinstance(result, str):
            # エラーメッセージをそのまま返す
            logger.warning("[change_search_space] %s", result)
            return {
                "messages": [ToolMessage(content=result, tool_call_id=tool_call_id)]
            }
        description = _describe_param_space(result)
        logger.info("[change_search_space] 探索空間を更新:\n%s", description)
        return {
            "messages": [
                ToolMessage(
                    content=f"探索空間を更新しました:\n{description}",
                    tool_call_id=tool_call_id,
                )
            ],
            "current_param_space": result,
        }

    def _should_continue(self, state: SupervisorState) -> Literal["continue", "end"]:
        """次のステップを決定する条件分岐。"""
        if state.remaining_trials <= 0:
            return "end"
        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"
        return "end"

    def _find_best(
        self, trial_records: list[TrialRecord]
    ) -> tuple[float, dict[str, Any]]:
        """試行履歴から最良スコアとパラメータを返す。"""
        if not trial_records:
            return 0.0, {}
        best = max(trial_records, key=lambda r: r.score)
        return best.score, dict(best.params)

    def _build_result(
        self, final_state: dict[str, Any] | SupervisorState, config: HPOConfig
    ) -> HPOResult:
        """最終状態から HPOResult を構築する。"""
        # LangGraph の invoke は dict を返す場合がある
        if isinstance(final_state, dict):
            trial_records: list[TrialRecord] = final_state.get("trial_records", [])
        else:
            trial_records = final_state.trial_records

        best_score, best_params = self._find_best(trial_records)

        if trial_records:
            records_dicts = [r.to_dict() for r in trial_records]
            trials_df = pd.DataFrame(records_dicts)
        else:
            trials_df = pd.DataFrame()

        final_report = self._report_generator.generate_final(
            trial_records=trial_records,
            best_params=best_params,
            best_score=best_score,
            llm=self._llm,
            seed=config.seed,
            generated_param_space=self._generated_param_space,
        )

        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            trials_df=trials_df,
            report=final_report,
        )
