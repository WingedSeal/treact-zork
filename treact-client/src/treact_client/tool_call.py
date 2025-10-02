from dataclasses import dataclass
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
