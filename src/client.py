from mcp import ClientSession, StdioServerParameters, stdio_client
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
import pprint
import os

if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./access_key.json"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

server_params = StdioServerParameters(
    command="uv",
    args=["run", "mcp_server.py"],
)


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_react_agent(model=llm, tools=tools)
            response = await agent.ainvoke(
                {"messages": HumanMessage(content="Solve the zork game")}
            )
            pprint.pp(response)


if __name__ == "__main__":
    asyncio.run(main())
