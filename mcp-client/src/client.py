from mcp import ClientSession
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import pprint
import os
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv


load_dotenv("./mcp-client/.env")


if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./mcp-client/access_key.json"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)


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

    async def talk_with_zork(self):
        agent = create_react_agent(model=llm, tools=self.tools, debug=True)
        agent_response = await agent.ainvoke(
            {
                "messages": "First time calling tool, just parse empty list. After that, win the game (Zork) by calling Zork api"
            },
            {"recursion_limit": 500},
        )
        return agent_response

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


# Global client instance
client = MCPClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await client.connect_to_server()
    print("MCP Client connected and ready!")
    yield
    # Shutdown
    await client.cleanup()
    print("MCP Client cleaned up!")


app = FastAPI(title="python application", lifespan=lifespan)


@app.post("/react/")
async def process_query():
    result = await client.talk_with_zork()
    return result


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("CLIENT_PORT")))
