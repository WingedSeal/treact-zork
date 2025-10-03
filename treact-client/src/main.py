import asyncio
from treact_client import run_client, AIMode


def react_client() -> None:
    asyncio.run(run_client(AIMode.REACT))


def treact_client() -> None:
    asyncio.run(run_client(AIMode.TREACT))


def standard_client() -> None:
    asyncio.run(run_client(AIMode.STANDARD))
