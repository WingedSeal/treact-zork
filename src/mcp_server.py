from mcp.server.fastmcp import FastMCP
import httpx
import pprint
from typing import List

mcp = FastMCP(name="Talk to zork", host="localhost", port=8500)


@mcp.tool(name="north")
def go_north(history: List[str]) -> dict:
    """Sends a command to the Zork server to go north.

    Arguments:
        history (List[str]): The history of commands sent to the Zork server.

    Returns:
        dict: The response from the Zork server.
    """

    try:
        result = httpx.post(
            url="http://localhost:8000/zork/zork285",
            json={"commands": history + ["go north"]},
        )
        pprint.pp(result.json())
        return result.json()
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


@mcp.tool(name="south")
def go_south(history: List[str]) -> dict:
    """Sends a command to the Zork server to go south.

    Arguments:
        history (List[str]): The history of commands sent to the Zork server.

    Returns:
        dict: The response from the Zork server.
    """

    try:
        result = httpx.post(
            url="http://localhost:8000/zork/zork285",
            json={"commands": history + ["go south"]},
        )
        pprint.pp(result.json())
        return result.json()
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
