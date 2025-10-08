from queue import Queue
from typing import TypeVar

T = TypeVar("T")


class PeekableQueue(Queue[T]):
    def peek(self) -> T:
        with self.mutex:
            return self.queue[0]
