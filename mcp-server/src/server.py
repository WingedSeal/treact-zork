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

history = []


# @mcp.tool(name="zork-api")
# def call_zork(command: str) -> dict:
#     """
#     Executes a single command in the Zork text adventure game with local history management.

#     This tool interfaces with a Zork game server by sending individual commands. The game
#     state is maintained locally on the client side, so each command builds upon the previous
#     game state without needing to replay the entire command history.

#     Arguments:
#         command (str): A single Zork game command to execute. This should be a valid game
#                       command such as movement ('north', 'south'), item interaction
#                       ('take lamp', 'drop sword'), actions ('open door', 'kill troll'),
#                       or utility commands ('look', 'inventory', 'score').

#     Returns:
#         dict: Server response containing the game's output after executing the command.
#               Typically includes room descriptions, item interactions, puzzle feedback,
#               combat results, or error messages for invalid commands.

#     Example Usage:
#         # First command in game
#         command = 'look'

#         # Movement command
#         command = 'north'

#         # Item interaction
#         command = 'take lamp'

#         # Each command executes on the current game state

#     Note: The game state persists between commands on the server side, so there's no need
#           to maintain or send command history. Each command operates on the current state
#           resulting from all previous commands executed in this session.
#     """

#     try:
#         history.append(command)
#         logger.info(f"Input: {history}")

#         result = httpx.post(
#             url="http://localhost:8000/zork/zork285",
#             json={"commands": history},
#             timeout=500,
# @mcp.tool(name="zork-285-api")
# def zork_285_api(history: list[str]) -> dict:
#     """Sends a new command to Zork by including all previous commands to get the final outcome.
#
#     Arguments:
#         history (list[str]): The history of previous commands sent to the Zork server plus a new command.
#
#     Returns:
#         dict: The response from the Zork server.
#     """
#
#     try:
#         logger.info(f"Input: {history}")
#         result = httpx.post(
#             url="http://localhost:8000/zork/zork285",
#             json={"commands": history},
#             timeout=300
#         )
#         if result.status_code == 200:
#             pprint.pp(result.json())
#             logger.info(result.json())
#             return result.json()
#         else:
#             raise Exception("cannot call Zork")
#     except httpx.HTTPError as e:
#         pprint.pp(e)
#         return {"response": f"Error: {str(e)}"}


@mcp.tool(name="zork-285-api-gen-key")
def zork_285_api_gen_key() -> dict:
    """
    Generate a new Zork Session a new key to access the session


    Returns:
        dict: Session Key and Zork's initial response.
    """

    try:
        result = httpx.get(url="http://localhost:8000/gen_key/zork285", timeout=300)
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())
            return result.json()
        else:
            raise Exception("cannot call Zork")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"key": "Error", "response": str(e)}


@mcp.tool(name="zork-285-api-use-key")
def zork_285_api_use_key(command: str, session_key: str) -> dict:
    """
        Using the session key obtained from "zork-285-api-gen-key", send new command to Zork


        Returns:
    #       dict: The response from the Zork server.
    """

    try:
        result = httpx.post(
            url="http://localhost:8000/use_key/zork285",
            json={"command": command, "key": session_key},
            timeout=300,
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
