from typing import TypedDict, NotRequired
from langchain_core.language_models import BaseChatModel
from .tool_call import ToolCall, ToolCallResult
from .ai_model_response import AIModelResponse


class State(TypedDict):
    llm: BaseChatModel
    template: str

    history: list[ToolCallResult]
    history_max_length: int

    tool_calls: list[ToolCall]
    last_result_content: str
    structured_response: AIModelResponse | None

    current_step: int
    maximum_step: int

    debug: bool
    key: str

    game: str

    missing_tool_calls: bool
    hallucinate_count: int
    hallucinate_count_threshold: int
    hallucinate_streak: int

    give_up: bool
