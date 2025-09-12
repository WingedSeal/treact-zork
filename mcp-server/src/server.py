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
                f"{log_dir}/mcp-server_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log",
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


@mcp.tool(name="api-gen-key")
def api_gen_key(game: str) -> dict:
    """
    Generate a new Session with a new key to access for the given game.

    Arguments:
        game: str: The game to start a new session

    Returns:
        dict: Session Key and game's initial response.
    """

    try:
        result = httpx.get(url=f"http://localhost:8000/gen_key/{game}", timeout=300)
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())
            return result.json()
        else:
            raise Exception(f"Cannot call api for {game}")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"key": "Error", "response": str(e)}


@mcp.tool(name="api-use-key")
def api_use_key(game: str, command: str, session_key: str) -> dict:
    """
    Using the session key obtained from "api-gen-key", send new command to the game.
    However, the key can only be used once. But using it generates a new key that can be used.

    Arguments:
        command (str): The command to send to the game.
        session_key (str): The session key obtained from "api-gen-key" or "api-use-key".

    Returns:
        dict: The response from the game server and a new session key.
    """

    try:
        logger.info({"Input command": command, "session_key": session_key})
        pprint.pp({"Input command": command, "session_key": session_key})
        result = httpx.post(
            url=f"http://localhost:8000/use_key/{game}",
            json={"command": command, "key": session_key},
            timeout=300,
        )
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())
            return result.json()
        else:
            raise Exception(f"Cannot call api for {game}")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


@mcp.tool(name="api-get-words")
def api_get_words(game: str) -> dict:
    """
    Get the list of all possible words from the given game.
    (Must be used only one time)

    Arguments:
        None

    Returns:
        dict: The list of all possible commands from the given game.

    Examples:
        {"words": ["take", "inventory", "north", "lamp", ...]}
    """

    try:
        result = httpx.get(
            url=f"http://localhost:8000/dict/{game}",
            params={"types": False},
            timeout=300,
        )
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())

            return result.json()
        else:
            raise Exception(f"Cannot call api for {game}")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


@mcp.tool(name="api-get-dict")
def api_get_dict(game: str) -> dict:
    """
    Get the dictionary of words from the given game.

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
            url=f"http://localhost:8000/dict_with_types/{game}",
            params={"types": True},
            timeout=300,
        )
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())
            return result.json()
        else:
            raise Exception(f"Cannot call api for {game}")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


@mcp.tool(name="api-get-chat-log")
def api_get_chat_log(game: str, session_key: str) -> dict:
    """
    Get the chat log of the current game session using the session key.

    Arguments:
        session_key (str): The session key obtained from "api-gen-key" or "api-use-key".

    Returns:
        dict: The chat log of the current game session.
    """

    try:
        result = httpx.get(
            url=f"http://localhost:8000/chat_log/{game}",
            params={"key": session_key},
            timeout=300,
        )
        if result.status_code == 200:
            pprint.pp(result.json())
            logger.info(result.json())
            with open(
                f"{log_dir}/chat_log_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"{result.json().get('log', '')}")

            return result.json()
        else:
            raise Exception(f"Cannot call api for {game}")
    except httpx.HTTPError as e:
        pprint.pp(e)
        return {"response": f"Error: {str(e)}"}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
