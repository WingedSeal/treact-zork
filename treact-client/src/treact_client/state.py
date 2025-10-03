from dataclasses import dataclass
from typing import TypedDict, NotRequired, Any
from langchain_core.language_models import BaseChatModel
from .tool_call import ToolCall, ToolCallResult
from .ai_model_response import AIModelResponse


@dataclass(kw_only=True, frozen=True)
class ModelSettings:
    llm: BaseChatModel
    prompt_template: str
    game_name: str
    maximum_step: int
    missing_tool_call_threshold: int
    history_max_length: int

    def to_new_state(self) -> "State":
        return State(
            {
                "model_settings": self,
                "current_step": 0,
                "missing_tool_call_count": 0,
                "tool_call_result_history": [],
                "tool_calls": [],
                "last_ai_message_result_content": "",
                "ai_model_response": None,
                "zork_session_key": "",
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


class State(_CSVLoggedState, TypedDict):
    model_settings: ModelSettings
    tool_call_result_history: list[ToolCallResult]
    tool_calls: list[ToolCall]
    last_ai_message_result_content: str | list[str | dict[Any, Any]]
    ai_model_response: AIModelResponse | None
    zork_session_key: str
    is_missing_tool_call: bool


class PartialState(TypedDict):
    current_step: NotRequired[int]
    missing_tool_call_count: NotRequired[int]

    tool_call_result_history: NotRequired[list[ToolCallResult]]
    tool_calls: NotRequired[list[ToolCall]]
    last_ai_message_result_content: NotRequired[str | list[str | dict[Any, Any]]]
    ai_model_response: NotRequired[AIModelResponse | None]
    zork_session_key: NotRequired[str]
    is_missing_tool_call: NotRequired[bool]
