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
    command_history: Annotated[
        list[str],
        Field(
            description="Complete chronological list of ALL Zork commands executed in this session, "
            "starting from the very first command and ending with the most recent command. "
            "Each string represents a single game command (e.g., 'look', 'take lamp', 'north'). "
            "This must include every command from game start to current state."
        ),
    ]
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
    next_action_planned: Annotated[
        str,
        Field(
            description="The specific next command you plan to execute in the Zork game. "
            "Should be a valid Zork command like 'north', 'take sword', 'examine door', etc. "
            "This ensures continuous gameplay progression."
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
        You are playing Zork, a text adventure game. Your goal is to **WIN** by collecting all 20 treasures.
        In the beginning, 'Welcome to adventure.\nYou are in an open field west of a big white house, with a closed, locked\nfront door.'
        
        ## YOUR REQUIRED STEPS ##
        STEP 1: Start the game by Calling Zork tool with proper command
        STEP 2: Read the game response
        STEP 3: Understand the current scenarion and come up with a new command
        STEP 4: Call Zork tool again with the new command
        STEP 5: Repeat steps 2-4 until victory
        
        ## CRITICAL RULES ##
        - NEVER analyze without acting first
        - Keep playing until you collect all 20 treasures

        ## COMMAND EXAMPLES ##
        First game start: 'look'
        Movement: 'north', 'south', 'east', 'west', 'up', 'down'  
        Items: 'take lamp', 'drop sack', 'inventory', 'examine object'
        Actions: 'open door', 'turn on lamp', 'kill troll', 'move rug'

        ## ERROR RECOVERY ##
        - "I don't know that word" → Try different phrasing
        - "I don't know how to do that" → Use different command
        - Darkness → Find light source ('take lamp', 'turn on lamp')
        - Blocked → Look for keys or alternative paths
        
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
    print(result["structured_response"].command_history)
    print(result["structured_response"].game_completed)

    await client.cleanup()
    return


if __name__ == "__main__":
    # uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("CLIENT_PORT")))
    asyncio.run(main())
