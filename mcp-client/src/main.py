from treact_client import run_client

import asyncio


def async_main():
    print("test1")
    asyncio.run(run_client())


if __name__ == "__main__":
    async_main()
