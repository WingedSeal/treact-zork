from queue import Queue
from typing import TypeVar, cast

T = TypeVar("T")


class PeekableQueue(Queue[T]):
    def peek(self) -> T:
        with self.mutex:
            return cast(T, self.queue[-1])
