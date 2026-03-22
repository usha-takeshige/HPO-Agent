"""LangGraph state definition for Supervisor."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict

from hpo_agent.models import HPOConfig, ParamSpace, TrialRecord


class SupervisorState(BaseModel):
    """LangGraph グラフの状態を保持する Pydantic モデル。

    Attributes:
        messages: LangGraph のメッセージリスト（add_messages リデューサーで管理）。
        trial_records: これまでに実施した試行のリスト。
        remaining_trials: 残りの試行回数。
        config: エージェント実行設定。
        current_report: 最新の中間レポート文字列。
        last_tool_reasoning: Supervisor が最後に出力したツール選択理由。
        current_param_space: narrow_search_space ツールによって更新された探索空間。
            None の場合は各ツールのデフォルト param_space を使用する。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[list[BaseMessage], add_messages]
    trial_records: list[TrialRecord] = []
    remaining_trials: int
    config: HPOConfig
    current_report: str = ""
    last_tool_reasoning: str = ""
    current_param_space: ParamSpace | None = None
