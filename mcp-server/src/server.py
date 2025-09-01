from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import httpx
import pprint
import logging
import datetime


log_dir = "./mcp-server_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                f"{log_dir}/mcp-server_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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


@mcp.tool(name="zork-285-api-gen-key")
def zork_285_api_gen_key() -> dict:
    """
    Generate a new Zork Session a new key to access the session

    Arguments:
        None

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
    Using the session key obtained from "zork-285-api-gen-key", send new command to Zork.
    However, the key can only be used once. But using it generate a new key that can be used.

    Arguments:
        command (str): The command to send to Zork.
        session_key (str): The session key obtained from "zork-285-api-gen-key" or "zork-285-api-use-key".

    Returns:
        dict: The response from the Zork server and a new session key.
    """

    try:
        logger.info({"Input command": command, "session_key": session_key})
        pprint.pp({"Input command": command, "session_key": session_key})
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


# @mcp.tool(name="zork-285-api-get-words")
# def zork_285_api_get_words() -> dict:
#     """
#     Get the list of all possible words from Zork game.
#     (Must be used only one time)

#     Arguments:
#         None

#     Returns:
#         dict: The list of all possible commands from Zork game.

#     Examples:
#         {"words": ["take", "inventory", "north", "lamp", ...]}
#     """

#     try:
#         result = httpx.get(
#             url=f"http://localhost:8000/dict/zork285",
#             params={"types": False},
#             timeout=300,
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


@mcp.tool(name="zork-285-api-get-dict")
def zork_285_api_get_dict() -> dict:
    """
    Get the dictionary of words from the Zork game.

    Arguments:
        None

    Returns:
        dict: The game dictionary includes all recognizable words from the game's parser,
            including commands, objects, directions, adjectives, and other vocabulary
            that the game engine can understand and process.

    Examples:
     { "dictonary": [
                {"word": "take", "word_types": ["verb"]},
                {"word": "lamp", "word_types": ["noun"]},
                {"word": "north", "word_types": ["direction"]},
                ...
                    ]
    }

    """

    try:
        result = httpx.get(
            url=f"http://localhost:8000/dict/zork285",
            params={"types": False},
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
