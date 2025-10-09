from dataclasses import dataclass, field
from typing import Any

from treact_client.log import get_logger

ToolServerResponse = dict[str, Any]
"""Response JSON from tool server"""

AI_RESULT_CONTENT = str | list[str | dict[Any, Any]]

logger = get_logger(__name__)


@dataclass(kw_only=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, str]
    ai_thought: AI_RESULT_CONTENT | None = field(default=None, repr=False)
    """Why did the AI decide to call this tool"""


@dataclass(kw_only=True)
class ToolCallResult(ToolCall):
    tool_server_response: ToolServerResponse


@dataclass
class ToolCallResultNode:
    tool_call_result: ToolCallResult
    parent_node: "ToolCallResultNode | None" = field(repr=False)

    def get_history(self, history_max_length: int) -> list[ToolCallResult]:
        history: list[ToolCallResult] = []
        current_node: ToolCallResultNode | None = self
        for _ in range(history_max_length):
            if current_node is None:
                break
            history.append(current_node.tool_call_result)
            current_node = current_node.parent_node

        return history[::-1]


class ToolCallResultNodeUpdate:
    class BaseUpdate:
        pass

    @dataclass
    class Pop(BaseUpdate):
        pass

    @dataclass
    class PutBack(BaseUpdate):
        items: list[ToolCallResultNode]
