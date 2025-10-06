from time import time
import random
import uuid

KEY_LENGTH = 4
MAX_SEED = 1000000


class KeyData:
    last_query_time: float
    command: str | None
    last_key: str
    seed: str

    def __init__(self, command: str | None, last_key: str, seed: str) -> None:
        self.last_query_time = time()
        self.command = command
        self.last_key = last_key
        self.seed = seed


class KeyManager:
    def __init__(self, games: list[str]) -> None:
        self.keys_data: dict[str, dict[str, KeyData]] = {game: {} for game in games}

    def __get_key(
        self, game: str, old_key: str, new_command: str | None, old_seed: str
    ) -> tuple[str, str]:
        while True:
            key = str(uuid.uuid4()).replace("-", "")[:KEY_LENGTH]
            if not key in self.keys_data[game]:
                break
        self.keys_data[game][key] = KeyData(new_command, old_key, old_seed)
        return key, self.keys_data[game][key].seed

    def gen_key(self, game: str) -> tuple[str, str]:
        seed = str(random.randint(1, MAX_SEED))
        return self.__get_key(game, "", None, seed)

    def add_command(self, game: str, key: str, command: str) -> tuple[str, str]:
        if key not in self.keys_data[game]:
            return "", ""
        new_key, seed = self.__get_key(game, key, command, self.get_seed(game, key))
        return new_key, seed

    def verify_key(self, game: str, key: str) -> bool:
        return key in self.keys_data[game]

    def get_seed(self, game: str, key: str) -> str:
        return self.keys_data[game][key].seed

    def get_history(self, game: str, key: str) -> list[str]:
        command_history: list[str] = []
        while True:
            key_data = self.keys_data[game][key]
            command = key_data.command
            if command is None:
                break
            command_history.append(command)
            key = key_data.last_key

        return command_history[::-1]


key_example = str(uuid.uuid4())[:KEY_LENGTH]
