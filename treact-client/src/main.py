import asyncio
from treact_client import run_client


def react_client() -> None:
    asyncio.run(run_client())


def treact_client() -> None:
    print("WIP")


def standard_client() -> None:
    pass
