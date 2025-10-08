import csv
from typing import TypedDict, cast

from langchain_core.language_models import BaseChatModel

from .ai_model_response import AIModelResponseTypes
from .log import LOG_DIRECTORY, get_current_time_string, get_logger
from .state import State, _CSVLoggedState

logger = get_logger(__name__)


class CSVFields(_CSVLoggedState, TypedDict):
    game_completed: AIModelResponseTypes.game_completed
    current_inventory: AIModelResponseTypes.current_inventory
    current_status: AIModelResponseTypes.current_status
    score: AIModelResponseTypes.score
    move: AIModelResponseTypes.move


class CSVLogger:
    def __init__(self, model: BaseChatModel, client_name: str) -> None:
        current_time_string = get_current_time_string()
        self.csv_file = (
            LOG_DIRECTORY
            / f"{client_name.lower().replace(' ', '')}_{model.__class__.__name__}_{current_time_string}.csv"
        )
        with self.csv_file.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSVFields.__annotations__.keys())
            writer.writeheader()
        logger.debug(f"Initialized CSVLogger at {self.csv_file.resolve().as_posix()}")

    def add_result_state(self, result_state: State) -> None:
        logger.debug(f"Adding result state")
        ai_model_response = result_state["ai_model_response"]
        assert ai_model_response is not None
        csv_row = cast(
            CSVFields,
            {
                **{
                    key: result_state[key]  # type: ignore
                    for key in _CSVLoggedState.__annotations__.keys()
                },
                **ai_model_response.model_dump(),
            },
        )
        with self.csv_file.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSVFields.__annotations__.keys())
            writer.writerow(csv_row)
