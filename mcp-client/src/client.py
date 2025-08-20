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
        You are playing Zork, a text adventure game. Your goal is to WIN by collecting all 20 treasures.

        COMMAND HISTORY: {messages}

        ## MANDATORY FIRST ACTION ##
        You MUST immediately call the Zork tool with this exact input: {messages}

        If the command history is [] (empty list), you MUST call the Zork tool with: []
        If the command history has commands, you MUST call the Zork tool with those exact commands.

        ## YOUR REQUIRED STEPS ##
        STEP 1: Call Zork tool with input: {messages}
        STEP 2: Read the game response
        STEP 3: Add ONE new command to the history
        STEP 4: Call Zork tool again with the expanded history
        STEP 5: Repeat steps 2-4 until victory

        ###Important###
        DO NOT MODIFY COMMAND HISTORY
        
        ## CRITICAL RULES ##
        - Your FIRST action must be calling the Zork tool
        - NEVER skip the first tool call
        - NEVER analyze without acting first
        - Each response MUST include at least one Zork tool call
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
    count = 0
    complete = False
    final_result = []
    while count <= 10 and (not complete):
        result = await client.talk_with_zork(history=final_result, llm=llama)
        pprint.pp(result["structured_response"])
        print(result["structured_response"].command_history)
        print(result["structured_response"].game_completed)
        final_result = result["structured_response"].command_history
        complete = result["structured_response"].game_completed
        count += 1
    print(final_result)
    await client.cleanup()
    return


if __name__ == "__main__":
    # uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("CLIENT_PORT")))
    asyncio.run(main())
