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
import time

load_dotenv("./mcp-client/.env")

test_result = "./test_result"
if not os.path.exists(test_result):
    os.makedirs(test_result)


api_key = os.getenv("API_KEY")

if not api_key:
    model = ChatOllama(model="llama3.1", temperature=0)
else:
    os.environ["GOOGLE_API_KEY"] = api_key
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        # max_output_tokens=8000,
    )


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
    llm: Any
    template: str

    history: list[Any]
    tool_calls: list[Any]
    structured_response: Any

    current_step: int
    maximum_step: int

    empty_call_threshold: int
    max_empty_call_threshold: int

    debug: bool
    key: str


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
            empty_call_threshold = state["empty_call_threshold"]
            if state["current_step"] == 0:
                final_result = [
                    {
                        "name": "zork-1-api-get-words",
                        "args": {},
                        "type": "tool_call",
                    }
                ]

            elif state["current_step"] == 1:
                final_result = [
                    {
                        "name": "zork-1-api-gen-key",
                        "args": {},
                        "type": "tool_call",
                    }
                ]
            elif state["current_step"] >= state["maximum_step"] - 1:
                final_result = [
                    {
                        "name": "zork-1-api-use-key",
                        "args": {"command": "score", "session_key": state["key"]},
                        "type": "tool_call",
                    }
                ]
            else:
                llm = state["llm"]
                llm_with_tools = llm.bind_tools(self.tools)
                template = state["template"]
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm_with_tools
                result = await chain.ainvoke(
                    {
                        "history": state["history"],
                        "current_step": state["current_step"],
                        "maximum_step": state["maximum_step"],
                    }
                )
                # time.sleep(7)
                final_result = result.tool_calls

            if not final_result:
                empty_call_threshold += 1
                print(
                    f"**Empty tool call {empty_call_threshold}**: use command 'score'"
                )
                final_result = [
                    {
                        "name": "zork-1-api-use-key",
                        "args": {"command": "score", "session_key": state["key"]},
                        "type": "tool_call",
                    }
                ]

            if state["debug"]:
                print(f"Count: {state['current_step'] + 1}")
                pprint.pp(final_result)

            return {
                "tool_calls": final_result,
                "current_step": state["current_step"] + 1,
                "empty_call_threshold": empty_call_threshold,
            }

        def should_continue(state: State):
            if state["empty_call_threshold"] > state["max_empty_call_threshold"]:
                if state["debug"]:
                    pprint.pp(f"Reach empty_call_threshold")
                return "END"
            elif state["current_step"] > state["maximum_step"]:
                if state["debug"]:
                    pprint.pp(f"Reach Maximum")
                return "END"
            elif state["tool_calls"]:
                if state["debug"]:
                    pprint.pp("Action")
                return "Action"
            else:
                return "END"

        async def call_tools(state: State):
            assert self.session is not None
            total_result = []
            key = ""
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
                    total_result.append(
                        {
                            "tool_name": tool["name"],
                            "arguments": tool["args"],
                            "resposne": final_result,
                        }
                    )
                    key = final_result["new_key"]
            except Exception as e:
                total_result.append({"response": f"Error: {str(e)}"})
            total_result = state["history"] + total_result
            return {"history": total_result, "key": key}

        async def summarize(state: State):
            llm_with_structured = state["llm"].with_structured_output(response)
            try:
                template = """Summarize the following history of commands and responses from playing zork
                
                            ### History ###
                            {history}
                
                            """

                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm_with_structured
                result = await chain.ainvoke({"history": state["history"][-10:-1]})
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
            case "react_test":
                template = """ You are playing zork 285 via accessing MCP tool
                You must follow the instruction
                Step 1: calling zork-285-api-gen-key tool to get the first generated key
                Step 2: Use the generated key with the proper command for playing zork as parameters for zork-285-api-use-key tool
                Step 3: Recieve the response with the generated key and proceed the Step 2 again
                
                Continue Step 3 until you collect all 20 treasures
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
                
                ### Important ###
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
                    
                    ### Important ###
                    - Keep tracking the game state and your inventory and also history commands
                    - Keep tracking the score and current steps
                    - when the current step : {current_step} equals to the maximum number of steps: {maximum_step}, you must quit the game
                    - When encountering troll, avoid the fight and find the sword first.
                    - Do not repeat the same action for more than 3 times.
                    - Do not stop until the current step equals the mximum number of steps

                    """
                agent_response = await self.agent.ainvoke(
                    {
                        "structured_response": None,
                        "template": template,
                        "history": [],
                        "llm": model,
                        "tool_calls": [],
                        "current_step": 0,
                        "maximum_step": 20,
                        "debug": debug,
                        "key": "",
                        "max_empty_call_threshold": 3,
                        "empty_call_threshold": 0,
                    },
                    config={"recursion_limit": 300},
                )
            case "test_implement":
                template = """ You are playing Zork (text-based game) via accessing MCP tool
                
                        Goal: collect as many treasures as possible.
                        
                        ### Previous result ###
                        {history}

                        ### Instruction ###
                            - Based on the previous result, use the previous generated key and assign proper command to play zork
                            
                        ## Important ##
                            - Do not repeate the same command
                            - Do not stop and keep playing
                        
                        """

                agent_response = await self.agent.ainvoke(
                    {
                        "structured_response": None,
                        "template": template,
                        "history": [],
                        "llm": model,
                        "tool_calls": [],
                        "current_step": 0,
                        "maximum_step": 400,
                        "debug": debug,
                        "key": "",
                        "max_empty_call_threshold": 5,
                        "empty_call_threshold": 0,
                    },
                    config={"recursion_limit": 800},
                    # Recurstion limit should be > 2 * maximum step
                )
            case _:
                raise Exception("Please assign the type of prompting")

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
    current = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    fields = [
        "game_completed",
        "current_status",
        "score",
        "current_step",
        "maximum_step",
    ]
    with open(
        f"{test_result}/result_{model.model.replace('/','-').replace(':', '-')}_{current}.csv",
        "w",
        newline="",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
    print(f"Model: {model.model}")
    try:
        history = []
        category = "test_implement"
        react = ["react_framework", "react_test"]
        debug = True
        for i in tqdm(iterable=range(2)):
            result = await client.talk_with_zork(
                history=history, model=model, category=category, debug=debug
            )
            final_result = result["structured_response"].model_dump()
            for key, value in final_result.items():
                print(f"{key} : {value}")

            if category not in react:
                print(f"current_step : {result['current_step']}")
                print(f"maximum_step : {result['maximum_step']}")
                final_result["current_step"] = result["current_step"]
                final_result["maximum_step"] = result["maximum_step"]

            test = [final_result]
            pprint.pp(f"Iteration {i+1} completed.")
            with open(
                f"{test_result}/result_{model.model.replace('/','-').replace(':', '-')}_{current}.csv",
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
