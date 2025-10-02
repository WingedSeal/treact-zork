from pathlib import Path
from typing import Iterable, cast
from .key_manager import KEY_LENGTH, KeyManager, key_example
import uvicorn
from fastapi import FastAPI
from .zork_api import ZorkInstance
from pydantic import BaseModel, Field
from .zork_dict import extract_dictionary_from_file
import os
import logging
import datetime


log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                f"{log_dir}/zork_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log",
                mode="a",
                encoding="utf-8",
            )
        ],
    )

    logger = logging.getLogger(__name__)
    return logger


logger = setup_logger()


GAME_DIRECTORY = "games/"


class CommandRequest(BaseModel):
    command: str = Field(description="Command to execute in Zork", examples=["look"])
    key: str = Field(
        description=f"{KEY_LENGTH} characters string for accessing Zork session",
        examples=[key_example],
    )


class CommandsRequest(BaseModel):
    commands: list[str] = Field(
        default=[],
        description="List of commands to execute in Zork",
        examples=[["look", "go north", "take lamp"]],
    )


def zork_post(commands: list[str], zork_file: str, seed: str) -> str:
    if not commands:
        with ZorkInstance(GAME_DIRECTORY + zork_file, seed) as zork:
            return zork.initial_response
    with ZorkInstance(GAME_DIRECTORY + zork_file, seed) as zork:
        response = ""
        for command in commands:
            response = zork.send_command(command)
    return response


def zork_history(commands: list[str], zork_file: str, seed: str) -> Iterable[str]:
    with ZorkInstance(GAME_DIRECTORY + zork_file, seed) as zork:
        yield zork.initial_response
        for command in commands:
            yield zork.send_command(command)


GAMES = {"zork285": "zork_285.z5", "zork1": "zork_1.z3"}
app = FastAPI()
app.state.key_manager = KeyManager(list(GAMES.keys()))


class GenKeyResponse(BaseModel):
    initial_response: str
    new_key: str


class UseKeyResponse(BaseModel):
    key_valid: bool
    response: str
    new_key: str


class GetDictResponse(BaseModel):
    dictionary: list[str]


class _WordWithTypes(BaseModel):
    word: str
    word_types: list[str]


class GetDictWithTypesResponse(BaseModel):
    dictionary: list[_WordWithTypes]


class ChatLogResponse(BaseModel):
    log: str


def create_endpoint(game: str, game_file: str):
    @app.get(f"/gen_key/{game}", response_model=GenKeyResponse)
    def gen_key():
        logger.info("Gen Key")
        key, seed = cast(KeyManager, app.state.key_manager).gen_key(game)
        logger.info(f"Generated key: {key} for game: {game} with seed {seed}")
        return {"initial_response": zork_post([], game_file, seed), "new_key": key}

    @app.post(f"/use_key/{game}", response_model=UseKeyResponse)
    def use_key(request: CommandRequest):
        logger.info("Use Key")
        logger.info(f"Using key: {request.key} for game: {game}")
        if request.command.lower() == "quit":
            return {
                "key_valid": True,
                "response": "You are not allowed to quit. Use the old key.",
                "new_key": request.key,
            }
        key_manager = cast(KeyManager, app.state.key_manager)
        new_key, seed = key_manager.add_command(game, request.key, request.command)
        logger.info(f"New Key: {new_key} and Seed: {seed}")
        if not new_key:
            return {"key_valid": False, "response": "", "new_key": ""}
        history = key_manager.get_history(game, new_key)
        logger.info(f"History for new key: {history}")
        return {
            "key_valid": True,
            "response": zork_post(history, game_file, seed),
            "new_key": new_key,
        }

    @app.get(f"/dict/{game}", response_model=GetDictResponse)
    def get_dict():
        zdict = extract_dictionary_from_file(Path(GAME_DIRECTORY + game_file))
        logger.info(f"Extracted dictionary {zdict} for game {game}")
        return {"dictionary": [word for word, word_types in zdict]}

    @app.get(f"/dict_with_types/{game}", response_model=GetDictWithTypesResponse)
    def get_dict_with_types():
        zdict = extract_dictionary_from_file(Path(GAME_DIRECTORY + game_file))
        logger.info(f"Extracted dictionary {zdict} for game {game} with types")
        return {
            "dictionary": [
                {"word": word, "word_types": word_types} for word, word_types in zdict
            ]
        }

    @app.get(f"/chat_log/{game}", response_model=ChatLogResponse)
    def get_chat_log(key: str):
        key_manager = cast(KeyManager, app.state.key_manager)
        if not key_manager.verify_key(game, key):
            return {"log": "INVALID_KEY"}
        command_history = key_manager.get_history(game, key)
        response_history = list(
            zork_history(command_history, game_file, key_manager.get_seed(game, key))
        )
        log = (
            response_history.pop(0)
            + "\n"
            + "\n".join(
                f" > {command}\n{response}"
                for command, response in zip(
                    command_history, response_history, strict=True
                )
            )
        )
        logger.info(f"Chat Log requested: {key}\n\n{log}")
        return {"log": log}


for game, game_file in GAMES.items():
    create_endpoint(game, game_file)

def run_server():
    uvicorn.run(app, host="localhost", port=8000)

