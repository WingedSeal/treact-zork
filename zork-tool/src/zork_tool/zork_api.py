import subprocess
import re
from queue import Queue, Empty
from threading import Thread
from time import perf_counter
from pathlib import Path

ANSI_STRIPPER = re.compile(r"\x1b[\[\(]\??[0-9;]*[a-zA-Z]|\x1b=")

FROTZ = "frotz"


class ZorkException(Exception):
    pass


class Zork:
    __slots__ = (
        "zork_file",
        "process",
        "reader_thread",
        "output_queue",
        "initial_response",
    )

    def __init__(self, zork_file: str, seed: str | None = None) -> None:
        self.zork_file = zork_file
        if not Path(zork_file).is_file():
            raise ZorkException(f"{zork_file} is not a file.")
        if not Zork.is_frotz_exist():
            raise ZorkException("'frotz' command not found.")

        if seed is None:
            args = [FROTZ, self.zork_file]
        else:
            args = [FROTZ, "-s", seed, self.zork_file]
        try:
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                universal_newlines=True,
            )

        except Exception as error:
            raise ZorkException("Failed to start the game") from error

        self.reader_thread = Thread(target=self.__read_output, daemon=True)
        self.output_queue: Queue[str] = Queue()
        self.reader_thread.start()
        self.initial_response = self._get_output()

    @staticmethod
    def is_frotz_exist():
        try:
            result = subprocess.run(
                [FROTZ, "--help"], capture_output=True, timeout=5)
            if "frotz" in result.stdout.decode().lower():
                return True
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return False
        return False

    def __read_output(self):
        while self.process and self.process.poll() is None:
            stdout = self.process.stdout
            if stdout is None:
                break
            line = stdout.readline()
            if not line:
                break
            line = ANSI_STRIPPER.sub("", line)
            self.output_queue.put(line.strip())

    def _get_output(
        self, timeout_second: float = 10, *, input_to_strip: str | None = None
    ) -> str:
        output_lines: list[str] = []
        start_time = perf_counter()

        while perf_counter() - start_time < timeout_second:
            try:
                line = self.output_queue.get(timeout=0.1)
                if input_to_strip is not None and line.startswith(">"):
                    while line.startswith(">"):
                        line = line.lstrip(">").lstrip()
                    if line.startswith(input_to_strip):
                        line = line[len(input_to_strip):].lstrip()
                    if not line:
                        continue
                output_lines.append(line)

            except Empty:
                if output_lines:
                    break
                continue

        return "\n".join(output_lines)

    def send_command(self, command: str) -> str:
        if not self.process or self.process.poll() is not None:
            raise ZorkException(
                "Sending command when the game is no longer running")
        if self.process is None:
            raise ZorkException("Process not found")
        stdin = self.process.stdin
        if stdin is None:
            raise ZorkException("stdin is None")
        stdin.write(command + "\n")
        stdin.flush()

        response = self._get_output(input_to_strip=command)
        return response

    def close(self):
        try:
            self.send_command("quit")
            self.send_command("y")
            self.process.wait(timeout=3)
        except:
            self.process.terminate()
        if self.process.poll() is None:
            self.process.kill()


class ZorkInstance:
    __slots__ = ("zork",)

    def __init__(self, zork_file: str, seed: str | None = None) -> None:
        self.zork = Zork(zork_file, seed)

    def __enter__(self):
        return self.zork

    def __exit__(self, exc_type, exc_value, traceback):
        self.zork.close()
