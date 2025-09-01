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
import orjson
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import csv
import datetime
from tqdm import tqdm
from mcp.types import TextContent

load_dotenv("./mcp-client/.env")

test_result = "./test_result"
if not os.path.exists(test_result):
    os.makedirs(test_result)

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


class State(TypedDict):
    structured_response: Any
    history: list[Any]
    llm: Any
    tool_calls: list[Any]
    current_step: int
    maximum_step: int
    debug: bool


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

        async def llm_call(state: State):
            llm = state["llm"]
            llm_with_tools = llm.bind_tools(self.tools)
            template = """
        
            You are playing Zork, a text adventure game. Your goal is to collect as many treasures as possible.

            ### Current History of tool calls and output ###
            {history}

            ## YOUR REQUIRED STEPS ##
            STEP 1: Start the game by calling zork-285-api-gen-key tool to get the generated key (first time only)
            STEP 2: call zork-285-api-get-dict tool to get the game dictionary (first time only)
            STEP 3: Use the generated key with the proper command as parameters for zork-285-api-use-key tool
            STEP 4: Read the game response
            STEP 5: Then call Zork tool again with the new command and the new generated key
            STEP 6: Repeat steps 3-5 for {maximum_step} times, then proceed to quit the game with the current status

            ## How to track current score ##
            - Input command 'score' to get the current score
            
            ## How to quit the game ##
            - Input command 'quit' to get the current score and confirmation whether to quit the game by calling tool
            
            ### Important ###
            - The first command should be 'help' to understand the game mechanics and available commands.
            - Keep tracking the game state and your inventory and also history commands
            - Keep tracking the score and current steps
            - when the current step : {current_step} equals to the maximum number of steps: {maximum_step}, you must quit the game
            - When encountering troll, avoid the fight and find the sword first.
            - Do not repeat the same action for more than 3 times.
            - Do not stop until the current step equals the mximum number of steps

            """
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | llm_with_tools
            result = await chain.ainvoke(
                {
                    "history": state["history"],
                    "current_step": state["current_step"],
                    "maximum_step": state["maximum_step"] - 3,
                }
            )
            count = state["current_step"] + 1
            if state["debug"]:
                print(f"Count: {count}")
                pprint.pp(result.tool_calls)
            return {
                "tool_calls": result.tool_calls,
                "current_step": count,
            }

        def should_continue(state: State):
            if state["current_step"] > state["maximum_step"]:
                if state["debug"]:
                    pprint.pp(f"Reach Maximum")
                return "END"
            if state["tool_calls"]:
                if state["debug"]:
                    pprint.pp("Action")
                return "Action"
            else:
                return "END"

        async def call_tools(state: State):
            assert self.session is not None
            total_result = []
            try:
                for tool in state["tool_calls"]:
                    result = await self.session.call_tool(
                        tool["name"], arguments=tool["args"]
                    )
                    if state["debug"]:
                        pprint.pp(result)
                    content = result.content[0]
                    if not isinstance(content, TextContent):
                        raise Exception(
                            f"It is not TextContent since it is {type(content)}"
                        )
                    if state["debug"]:
                        pprint.pp(content)
                        pprint.pp(type(content))
                    final_result = orjson.loads(content.text)
                    total_result.append(final_result)
            except Exception as e:
                total_result.append({"response": f"Error: {str(e)}"})
            total_result = state["history"] + total_result
            return {"history": total_result}

        async def summarize(state: State):
            llm_with_structured = state["llm"].with_structured_output(response)
            try:
                result = await llm_with_structured.ainvoke(
                    [HumanMessage(content=state["history"])]
                )
                return {
                    "structured_response": result,
                    "current_step": state["current_step"] - 1,
                }
            except Exception as e:
                return {"structured_response": f"Error: {str(e)}"}

        graph = StateGraph(State)
        graph.add_node("llm_call", llm_call)
        graph.add_node("environment", call_tools)
        graph.add_node("summarize", summarize)

        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges(
            "llm_call",
            should_continue,
            {"Action": "environment", "END": "summarize"},
        )
        graph.add_edge("environment", "llm_call")
        graph.add_edge("summarize", END)
        agent = graph.compile()
        self.agent = agent

    async def talk_with_zork(
        self, history: list[str], model: Any, category: str, debug: bool
    ):
        match category:
            case "react_framework":
                template = """
                You are playing Zork, a text adventure game. Your goal is to collect as many treasures as possible.

                ## YOUR REQUIRED STEPS ##
                STEP 1: Start the game by calling zork-285-api-gen-key tool to get the generated key (first time only)
                STEP 2: call zork-285-api-get-dict tool to get the game dictionary (first time only)
                STEP 3: Use the generated key with the proper command as parameters for zork-285-api-use-key tool
                STEP 4: Read the game response
                STEP 5: Then call Zork tool again with the new command and the new generated key
                STEP 6: Repeat steps 3-5 for 20 times, then proceed to quit the game with the current status
                
                ## How to track current score ##
                - Input command 'score' to get the current score
                
                ## How to quit the game ##
                - Input command 'quit' to get the current score and confirmation whether to quit the game
                
                ### Important ###
                - The first command should be 'help' to understand the game mechanics and available commands.
                - Keep tracking the game state and your inventory and also history commands
                - Keep tracking the score and current steps
                - After 20 steps of calling tool, proceed to quit the game with the current status
                - When encountering troll, avoid the fight and find the sword first.

                """
                assert self.tools is not None

                agent = create_react_agent(
                    model=model,
                    tools=self.tools,
                    prompt=template,
                    response_format=response,
                    debug=debug,
                )

                agent_response = await agent.ainvoke(
                    input={"messages": str(history)},
                    config={"recursion_limit": 100},
                )
            case "react_implement":
                agent_response = await self.agent.ainvoke(
                    {
                        "structured_response": None,
                        "history": [],
                        "llm": model,
                        "tool_calls": [],
                        "current_step": 0,
                        "maximum_step": 20,
                        "debug": debug,
                    },
                    config={"recursion_limit": 300},
                )
            case _:
                agent_response = await self.agent.ainvoke(
                    {
                        "structured_response": None,
                        "history": [],
                        "llm": model,
                        "tool_calls": [],
                        "current_step": 0,
                        "maximum_step": 20,
                        "debug": debug,
                    },
                    config={"recursion_limit": 300},
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
    current = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fields = [
        "game_completed",
        "current_status",
        "score",
        "current_step",
        "maximum_step",
    ]
    model = gemini
    with open(
        f"{test_result}/result_{model.model.replace('/','-')}_{current}.csv",
        "w",
        newline="",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
    try:
        for i in tqdm(iterable=range(2)):
            result = await client.talk_with_zork(
                history=[], model=model, category="react_implement", debug=True
            )
            print(result["structured_response"].game_completed)
            print(result["structured_response"].current_status)
            print(result["structured_response"].score)
            print(result["current_step"])
            print(result["maximum_step"])
            test = [
                {
                    "game_completed": result["structured_response"].game_completed,
                    "current_status": result["structured_response"].current_status,
                    "score": result["structured_response"].score,
                    "current_step": result["current_step"],
                    "maximum_step": result["maximum_step"],
                }
            ]
            pprint.pp(f"Iteration {i+1} completed.")
            with open(
                f"{test_result}/result_{model.model.replace('/','-')}_{current}.csv",
                "a",
                newline="",
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writerows(test)
    except Exception as e:
        print(f"Error: {str(e)}")

    await client.cleanup()
    return


if __name__ == "__main__":
    # uvicorn.run(app, host=os.getenv("SERVER_IP"), port=int(os.getenv("CLIENT_PORT")))
    asyncio.run(main())
