import asyncio

from treact_client import AIMode, run_client


def react_client() -> None:
    asyncio.run(run_client(AIMode.REACT))


def treact_client() -> None:
    asyncio.run(run_client(AIMode.TREACT))


def standard_client() -> None:
    asyncio.run(run_client(AIMode.STANDARD, iterations=10))
