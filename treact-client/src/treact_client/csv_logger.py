from .ai_mode import AIMode
import csv
from typing import TypedDict, cast
from .log import LOG_DIRECTORY, get_current_time_string
from .ai_model_response import AIModelResponse
from .state import State, _CSVLoggedState
from langchain_core.language_models import BaseChatModel


class _AIModelResponseTypes:
    game_completed = type["AIModelResponse.game_completed"]
    current_state = type["AIModelResponse.current_status"]
    score = type["AIModelResponse.score"]


class CSVFields(_CSVLoggedState, TypedDict):
    game_completed: _AIModelResponseTypes.game_completed
    current_status: _AIModelResponseTypes.current_state
    score: _AIModelResponseTypes.score


class CSVLogger:
    def __init__(self, model: BaseChatModel, ai_mode: AIMode) -> None:
        current_time_string = get_current_time_string()
        self.csv_file = (
            LOG_DIRECTORY
            / f"{ai_mode.value}_{model.__class__.__name__}_{current_time_string}.csv"
        )
        with self.csv_file.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSVFields.__annotations__.keys())
            writer.writeheader()

    def add_result_state(self, result_state: State) -> None:
        structured_response = result_state["structured_response"]
        assert structured_response is not None
        csv_row = cast(
            CSVFields,
            {
                **{
                    key: result_state[key]
                    for key in _CSVLoggedState.__annotations__.keys()
                },
                **structured_response.model_dump(),
            },
        )
        with self.csv_file.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSVFields.__annotations__.keys())
            writer.writerow(csv_row)
