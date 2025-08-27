from mcp import ClientSession
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import pprint
import os
from typing import Optional, Any, Annotated
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from typing import Any


load_dotenv("./mcp-client/.env")


if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./mcp-client/access_key.json"

gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    # max_output_tokens=8000,
)

llama = ChatOllama(model="llama3.1", temperature=0)


class response(BaseModel):
    game_completed: Annotated[
        bool,
        Field(
            description="True if the Zork game has been successfully completed by collecting all 20 treasures "
            "and achieving victory. False if the game is still in progress or if the player has died. "
            "Only set to True when the game explicitly indicates victory/winning condition met."
        ),
    ]
    current_status: Annotated[
        str,
        Field(
            description="Current game state description including: current location, recent game response, "
            "inventory status, immediate objectives, or any important game feedback. "
            "This helps track progress and inform the next decision."
        ),
    ]

    score: Annotated[
        int,
        Field(
            description="Current score in the Zork game, representing the number of treasures collected so far."
        ),
    ]


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools: Optional[Any]

    async def connect_to_server(self):
        read_stream, write_stream, session_ID = (
            await self.exit_stack.enter_async_context(
                streamablehttp_client(
                    f"http://localhost:{os.getenv('SERVER_PORT')}/mcp"
                )
            )
        )

        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await self.session.initialize()

        tool_results = await load_mcp_tools(self.session)
        pprint.pp(tool_results)
        print("Connected to server with tools:", [tool.name for tool in tool_results])
        self.tools = tool_results

    async def talk_with_zork(self, history: list[str], llm: Any):

        template = """
        You are playing Zork, a text adventure game. Your goal is to collect as many treasures as possible.

        ## YOUR REQUIRED STEPS ##
        STEP 1: Start the game by calling zork-285-api-gen-key tool to get the generated key
        STEP 2: Use the generated key with the proper command as parameters for zork-285-api-use-key tool
        STEP 3: Read the game response
        STEP 4: Understand the current scenarion and come up with a new command
        STEP 5: Call Zork tool again with the new command and the new generated key
        STEP 6: Repeat steps 2-5 for 20 times, then proceed to quit the game with the current status
        
        ## How to track current score ##
        - Input command 'score' to get the current score
        
        ## How to quit the game ##
        - Input command 'quit' to get the current score and confirmation whether to quit the game
        
        ### Important ###
        - The first command should be 'help' to understand the game mechanics and available commands.
        - Keep tracking the game state and your inventory and also history commands
        - Keep tracking the score and current steps
        - After 20 steps of calling tool, proceed to quit the game with the current status

        """
        agent = create_react_agent(
            model=llm,
            tools=self.tools,
            prompt=template,
            response_format=response,
            debug=True,
        )

        agent_response = await agent.ainvoke(
            input={"messages": str(history)},
            config={"recursion_limit": 5000},
        )
        return agent_response

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


# Global client instance
client = MCPClient()


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     await client.connect_to_server()
#     print("MCP Client connected and ready!")
#     yield
#     # Shutdown
#     await client.cleanup()
#     print("MCP Client cleaned up!")


# app = FastAPI(title="python application", lifespan=lifespan)


# @app.post("/react/")
# async def process_query():
#     result = await client.talk_with_zork()
#     return result


async def main():
    await client.connect_to_server()

    result = await client.talk_with_zork(history=[], llm=gemini)
    pprint.pp(result["structured_response"])
    print(result["structured_response"].game_completed)
    print(result["structured_response"].current_status)
    print(result["structured_response"].score)

    await client.cleanup()
    return


if __name__ == "__main__":
    # uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("CLIENT_PORT")))
    asyncio.run(main())
