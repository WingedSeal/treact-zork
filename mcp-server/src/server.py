from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import httpx
import pprint
import logging
import datetime


log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                f"{log_dir}/chatbot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                mode="a",
                encoding="utf-8",
            )
        ],
    )

    logger = logging.getLogger(__name__)
    return logger


logger = setup_logger()


load_dotenv("./mcp-server/.env")

mcp = FastMCP(
    name="mcp-server", host=os.getenv("SERVER_IP"), port=os.getenv("SERVER_PORT")
)


@mcp.tool(name="zork-api")
def call_zork(history: list[str]) -> dict:
    """
    Executes commands in the Zork text adventure game by maintaining complete command history.

    This tool interfaces with a Zork game server, sending the entire sequence of commands
    from game start to present. The server processes all commands sequentially to reach
    the current game state, then returns the response to the latest command.

    Arguments:
        history (list[str]): Complete chronological list of all commands executed in this
                            game session, including the new command to execute. Each string
                            represents a single game command (e.g., 'look', 'north', 'take lamp').
                            The server replays all commands to maintain proper game state.

    Returns:
        dict: Server response containing the game's output after executing the final command.
            Typically includes room descriptions, item interactions, puzzle feedback,
            combat results, or error messages for invalid commands.

    Example Usage:
        # First command in game
        history = ['look']

        # Adding movement command
        history = ['look', 'north', 'take lamp']

        # Server processes entire sequence and returns response to 'take lamp'

    Note: Each call must include ALL previous commands to ensure the game state is
        correctly reconstructed on the server side.
    """

    try:
        logger.info(f"Input: {history}")
        result = httpx.post(
            url="http://localhost:8000/zork/zork285",
            json={"commands": history},
            timeout=500,
        )
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())
            return result.json()
        else:
            raise Exception("cannot call Zork")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
