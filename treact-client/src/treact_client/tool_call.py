from dataclasses import dataclass, field
from typing import Any

ToolServerResponse = dict[str, Any]
"""Response JSON from tool server"""


@dataclass(kw_only=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, str]


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
            history.append(self.tool_call_result)
            if current_node is None:
                break
            current_node = self.parent_node

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
