from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict, Annotated
from .utils import PeekableQueue

from langchain_core.language_models import BaseChatModel

from .ai_model_response import AIModelResponse
from .log import get_logger
from .tool_call import (
    ToolCall,
    ToolCallResult,
    ToolCallResultNode,
    ToolCallResultNodeUpdate,
)

logger = get_logger(__name__)


@dataclass(kw_only=True, frozen=True)
class ModelSettings:
    llm: BaseChatModel
    prompt_template: str
    game_name: str
    maximum_step: int
    missing_tool_call_threshold: int
    history_max_length: int
    max_branch_per_node: int

    def to_new_state(self) -> "State":
        logger.debug(f"Generating new state from model settings: {self}")
        return State(
            {
                "model_settings": self,
                "current_step": 0,
                "missing_tool_call_count": 0,
                "tool_call_result_history_tree": PeekableQueue(),
                "tool_calls": [],
                "tool_call_results": [],
                "tool_calls_parent": None,
                "last_ai_message_result_content": "",
                "ai_model_response": None,
                "is_missing_tool_call": False,
                "maximum_step": self.maximum_step,
                "missing_tool_call_threshold": self.missing_tool_call_threshold,
            }
        )


class _CSVLoggedState(TypedDict):
    current_step: int
    maximum_step: int
    missing_tool_call_count: int
    missing_tool_call_threshold: int


def update_tool_call_result_history(
    current_queue: PeekableQueue[ToolCallResultNode],
    update: ToolCallResultNodeUpdate.BaseUpdate,
) -> None:
    match update:
        case ToolCallResultNodeUpdate.Pop:
            current_queue.get_nowait()
        case ToolCallResultNodeUpdate.PutBack(items):
            for item in items:
                current_queue.put_nowait(item)


class State(_CSVLoggedState, TypedDict):
    model_settings: ModelSettings
    tool_call_result_history_tree: Annotated[
        PeekableQueue[ToolCallResultNode], update_tool_call_result_history
    ]
    tool_calls: list[ToolCall]
    tool_call_results: list[ToolCallResult]
    tool_calls_parent: ToolCallResultNode | None
    last_ai_message_result_content: str | list[str | dict[Any, Any]]
    ai_model_response: AIModelResponse | None
    is_missing_tool_call: bool


class StateUpdate(TypedDict):
    current_step: NotRequired[int]
    missing_tool_call_count: NotRequired[int]

    tool_call_result_history_tree: NotRequired[ToolCallResultNodeUpdate.BaseUpdate]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_results: NotRequired[list[ToolCallResult]]
    tool_calls_parent: NotRequired[ToolCallResultNode | None]
    last_ai_message_result_content: NotRequired[str | list[str | dict[Any, Any]]]
    ai_model_response: NotRequired[AIModelResponse | None]
    is_missing_tool_call: NotRequired[bool]
