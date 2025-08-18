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
def call_zork(history: list[str], command: str) -> dict:
    """Sends a new command to Zork by including all previous commands to get the final outcome.

    Arguments:
        history (list[str]): The history of commands sent to the Zork server.
        command str: The new command

    Returns:
        dict: The response from the Zork server.
    """

    try:
        logger.info(f"Input: {history+[command]}")
        result = httpx.post(
            url="http://localhost:8000/zork/zork285",
            json={"commands": history + [command]},
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
