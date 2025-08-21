from time import time
import uuid
from copy import deepcopy

KEY_LENGTH = 4


class KeyData:
    last_query_time: float
    command_history: list[str]
    last_key: str

    def __init__(self) -> None:
        self.last_query_time = time()
        self.command_history = []
        self.last_key = ""

    def add_command(self, command: str, last_key: str):
        self.last_query_time = time()
        self.command_history.append(command)
        self.last_key = last_key


class KeyManager:
    def __init__(self, games: list[str]) -> None:
        self.keys: dict[str, dict[str, KeyData]] = {game: {} for game in games}

    def gen_key(self, game: str, old_key_data: KeyData | None = None) -> str:
        while True:
            key = str(uuid.uuid4())[:KEY_LENGTH]
            if not key in self.keys[game]:
                break
        if old_key_data is not None:
            self.keys[game][key] = deepcopy(old_key_data)
        else:
            self.keys[game][key] = KeyData()
        return key

    def add_command(self, game: str, key: str, command: str) -> str:
        if key not in self.keys[game]:
            return ""
        old_key_data = self.keys[game][key]
        new_key = self.gen_key(game, old_key_data)
        self.keys[game][new_key].add_command(command, key)
        return new_key

    def get_history(self, game: str, key: str) -> list[str]:
        return self.keys[game][key].command_history


key_example = str(uuid.uuid4())[:KEY_LENGTH]
