from typing import TypedDict, NotRequired, Any
from langchain_core.language_models import BaseChatModel
from .tool_call import ToolCall, ToolCallResult
from .ai_model_response import AIModelResponse


class _CSVLoggedState(TypedDict):
    current_step: int
    maximum_step: int

    hallucinate_count: int
    hallucinate_count_threshold: int


class State(_CSVLoggedState, TypedDict):
    llm: BaseChatModel
    template: str

    history: list[ToolCallResult]
    history_max_length: int

    tool_calls: list[ToolCall]
    last_result_content: str | list[str | dict[Any, Any]]
    structured_response: AIModelResponse | None

    key: str

    game: str

    missing_tool_calls: bool


class PartialState(TypedDict):
    current_step: NotRequired[int]
    maximum_step: NotRequired[int]

    hallucinate_count: NotRequired[int]
    hallucinate_count_threshold: NotRequired[int]

    llm: NotRequired[BaseChatModel]
    template: NotRequired[str]

    history: NotRequired[list[ToolCallResult]]
    history_max_length: NotRequired[int]

    tool_calls: NotRequired[list[ToolCall]]
    last_result_content: NotRequired[str | list[str | dict[Any, Any]]]
    structured_response: NotRequired[AIModelResponse | None]

    key: NotRequired[str]

    game: NotRequired[str]

    missing_tool_calls: NotRequired[bool]
