import asyncio

from treact_client import AIMode, run_client


def standard_client() -> None:
    asyncio.run(run_client(AIMode.STANDARD, iterations=10))


def action_client() -> None:
    asyncio.run(run_client(AIMode.ACTION))


def react_client() -> None:
    asyncio.run(run_client(AIMode.REACT))


def treact_client() -> None:
    asyncio.run(run_client(AIMode.TREACT))
