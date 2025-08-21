from time import time
import uuid

KEY_LENGTH = 4


class KeyData:
    last_query_time: float
    command_history: list[str]

    def __init__(self) -> None:
        self.last_query_time = time()
        self.command_history = []

    def add_command(self, command: str):
        self.last_query_time = time()
        self.command_history.append(command)


class KeyManager:
    def __init__(self) -> None:
        self.keys: dict[str, dict[str, KeyData]]

    def gen_key(self, game: str) -> str:
        while True:
            key = str(uuid.uuid4())[:KEY_LENGTH]
            if not key in self.keys[game]:
                break
        self.keys[game][key] = KeyData()
        return key

    def add_command(self, game: str, key: str, command: str) -> bool:
        if key not in self.keys[game]:
            return False
        self.keys[game][key].add_command(command)
        return True

    def get_history(self, game: str, key: str) -> list[str]:
        return self.keys[game][key].command_history


key_example = str(uuid.uuid4())[:KEY_LENGTH]
