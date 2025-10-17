from dataclasses import dataclass
from queue import Empty
from typing import Annotated, NotRequired, TypedDict, TypeVar

from langchain_core.language_models import BaseChatModel

from .ai_model_response import AIModelResponse
from .log import get_logger
from .tool_call import (
    AI_RESULT_CONTENT,
    ToolCall,
    ToolCallResult,
    ToolCallResultNode,
    ToolCallResultNodeUpdate,
)
from .utils import PeekableQueue

logger = get_logger(__name__)


@dataclass(kw_only=True, frozen=True)
class ModelSettings:
    llm: BaseChatModel
    """BaseChatModel"""
    prompt_template: str
    """Prompt used as initial system message"""
    game_name: str
    """Zork's game name. This name must match the game name in the Zork server"""
    maximum_step: int
    """How many nodes the agent can traverse before aborting"""
    len_tool_calls_out_of_range_threshold: int
    """How many times can LLM call too many or too little tools in a row before aborting"""
    history_max_length: int
    max_branch_per_node: int
    min_tool_calls: int
    max_tool_calls: int

    def to_new_state(self) -> "State":
        logger.debug(f"Generating new state from model settings: {self}")
        return State(
            {
                "model_settings": self,
                "current_step": 0,
                "len_tool_calls_out_of_range_count": 0,
                "tool_call_result_history_tree": PeekableQueue(),
                "tool_calls": [],
                "tool_call_results": [],
                "tool_calls_parent": None,
                "last_ai_thought": None,
                "ai_thoughts_len_tool_calls_out_of_range": [],
                "ai_model_response": None,
                "maximum_step": self.maximum_step,
                "len_tool_calls_out_of_range_threshold": self.len_tool_calls_out_of_range_threshold,
            }
        )


class _CSVLoggedState(TypedDict):
    current_step: int
    maximum_step: int
    len_tool_calls_out_of_range_count: int
    len_tool_calls_out_of_range_threshold: int


T = TypeVar("T", bound=PeekableQueue[ToolCallResultNode])


def update_tool_call_result_history(
    current_queue: T,
    update: ToolCallResultNodeUpdate.BaseUpdate | T,
) -> T:
    match update:
        case ToolCallResultNodeUpdate.Pop():
            try:
                node = current_queue.get_nowait()
                logger.debug(f"Popping from history tree: {node}")
            except Empty:
                logger.info("Queue empty, nothing to pop")
        case ToolCallResultNodeUpdate.PutBack(items):
            for item in items:
                current_queue.put_nowait(item)
        case PeekableQueue():
            return update
        case _:
            raise NotImplementedError(f"ToolCallResultNodeUpdate.BaseUpdate not handled: {update}")
    return current_queue


class State(_CSVLoggedState, TypedDict):
    model_settings: ModelSettings
    """The constant settings for determining agent's behavior"""
    tool_call_result_history_tree: Annotated[
        PeekableQueue[ToolCallResultNode], update_tool_call_result_history
    ]
    """Depth first search queue for traversing history of tool calls' results"""
    tool_calls: list[ToolCall]
    """List of tool calls LLM is trying to perform"""
    tool_call_results: list[ToolCallResult]
    """Results of tool calls agent just perform"""
    tool_calls_parent: ToolCallResultNode | None
    """A node in tool calls' results history tree that the upcoming generated result will have as the parent"""
    ai_model_response: AIModelResponse | None
    """Response of the model summarizing everything"""
    ai_thoughts_len_tool_calls_out_of_range: list[tuple[int, AI_RESULT_CONTENT]]
    """The thought (result's content) of the LLM prior to calling too many or too little tools"""
    last_ai_thought: AI_RESULT_CONTENT | None
    """The last thought (result's content) of the LLM. Used for handling when calling too many or too little tools"""


class StateUpdate(TypedDict):
    current_step: NotRequired[int]
    len_tool_calls_out_of_range_count: NotRequired[int]

    tool_call_result_history_tree: NotRequired[ToolCallResultNodeUpdate.BaseUpdate]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_results: NotRequired[list[ToolCallResult]]
    tool_calls_parent: NotRequired[ToolCallResultNode | None]
    ai_model_response: NotRequired[AIModelResponse | None]
    ai_thoughts_len_tool_calls_out_of_range: NotRequired[
        list[tuple[int, AI_RESULT_CONTENT]]
    ]
    last_ai_thought: NotRequired[AI_RESULT_CONTENT]
