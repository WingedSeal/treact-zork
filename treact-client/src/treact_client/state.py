from typing import TypedDict, NotRequired, Any
from langchain_core.language_models import BaseChatModel
from .tool_call import ToolCall, ToolCallResult
from .ai_model_response import AIModelResponse


class _CSVLoggedState(TypedDict):
    current_step: int
    maximum_step: int

    missing_tool_call_count: int
    missing_tool_call_threshold: int


class State(_CSVLoggedState, TypedDict):
    llm: BaseChatModel
    prompt_template: str

    tool_call_result_history: list[ToolCallResult]
    history_max_length: int

    tool_calls: list[ToolCall]
    last_ai_message_result_content: str | list[str | dict[Any, Any]]
    ai_model_response: AIModelResponse | None

    zork_session_key: str
    game_name: str

    is_missing_tool_call: bool


class PartialState(TypedDict):
    current_step: NotRequired[int]
    maximum_step: NotRequired[int]

    missing_tool_call_count: NotRequired[int]
    missing_tool_call_threshold: NotRequired[int]

    llm: NotRequired[BaseChatModel]
    prompt_template: NotRequired[str]

    tool_call_result_history: NotRequired[list[ToolCallResult]]
    history_max_length: NotRequired[int]

    tool_calls: NotRequired[list[ToolCall]]
    last_ai_message_result_content: NotRequired[str | list[str | dict[Any, Any]]]
    ai_model_response: NotRequired[AIModelResponse | None]

    zork_session_key: NotRequired[str]
    game_name: NotRequired[str]

    is_missing_tool_call: NotRequired[bool]
