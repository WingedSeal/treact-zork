from gc import collect
from mcp import ClientSession
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
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
from langchain_core.messages import SystemMessage, AIMessageChunk, AIMessage
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
import json
import re
import trace
import math


load_dotenv("./mcp-client/.env")

test_result = "./logs/test_result"
if not os.path.exists(test_result):
    os.makedirs(test_result)


api_key = os.getenv("API_KEY")


if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        # max_output_tokens=8000,
    )
else:
    # model = ChatOllama(model="gpt-oss:20b", temperature=0)
    model = ChatOllama(model="qwen3:8b", temperature=0)
    # model = ChatOllama(model="llama3.1:latest", temperature=0)


# class response(BaseModel):
#     game_completed: Annotated[
#         bool,
#         Field(
#             description="True if the Zork game has been successfully completed by collecting all 20 treasures "
#             "and achieving victory. False if the game is still in progress or if the player has died. "
#             "Only set to True when the game explicitly indicates victory/winning condition met."
#         ),
#     ]
#     current_status: Annotated[
#         str,
#         Field(
#             description="Current game state description including: current location, recent game response, "
#             "inventory status, immediate objectives, or any important game feedback. "
#             "This helps track progress and inform the next decision."
#         ),
#     ]

#     score: Annotated[
#         int,
#         Field(
#             description="Current score in the Zork game, representing the number of treasures collected so far."
#         ),
#     ]


class State(TypedDict):
    llm: Any
    template: str

    history: list[Any]
    history_max_length: int

    tool_calls: list[Any]
    last_result_content: str
    structured_response: Any

    current_step: int
    maximum_step: int

    debug: bool
    key: str

    game: str

    missing_tool_calls: bool
    hallucinate_count: int
    hallucinate_count_threshold: int
    hallucinate_streak: int

    give_up: bool


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
            last_result_content = ""
            if state["current_step"] == 0:
                final_result = [
                    {
                        "name": "api-get-dict",
                        "args": {"game": state["game"]},
                        "type": "tool_call",
                    }
                ]

            elif state["current_step"] == 1:
                final_result = [
                    {
                        "name": "api-gen-key",
                        "args": {"game": state["game"]},
                        "type": "tool_call",
                    }
                ]
            else:

                assert self.tools is not None

                llm: ChatGoogleGenerativeAI | ChatOllama = state["llm"]
                llm_with_tools = llm.bind_tools(self.tools)

                template = state["template"]
                prompt = ChatPromptTemplate.from_template(template)

                # Handle missing tool calls by appending a system message to force calling an action
                if state["missing_tool_calls"]:
                    state["missing_tool_calls"] = False
                    prompt.append(
                        SystemMessage(
                            content="You are not calling any tools, and the game has not yet ended. You put your command in **Actions** but forgot to also put it in tool calls. Call some tools."
                        )
                    )
                chain = prompt | llm_with_tools

                result = AIMessageChunk(content="")
                print("-- THINKING TIME --\n")
                async for chunk in chain.astream(
                    {
                        "history": state["history"],
                        "current_step": state["current_step"],
                        "maximum_step": state["maximum_step"],
                    }
                ):
                    tool_calls = getattr(chunk, "tool_calls", None)
                    if tool_calls:
                        print(f"\n[Tool Call]: {tool_calls}", flush=True)
                    elif getattr(chunk, "content", None):
                        print(chunk.content, end="", flush=True)
                    result += chunk
                print("\n-- DONE THINKING --\n")

                if state.get("debug", False):
                    print(f"Step = {state['current_step']}")
                    print("\nFull LLM Response:")
                    print(f"Content: {result.content}")
                    print(
                        f"Response Metadata: {getattr(result, 'response_metadata', {})}"
                    )
                    print(f"Tool Calls: {getattr(result, 'tool_calls', [])}")

                final_result = getattr(result, "tool_calls", [])
                last_result_content = result.content

            return {
                "tool_calls": final_result,
                "last_result_content": last_result_content,
                "current_step": state["current_step"] + 1,
            }

        def hallucinate_counter_measure(state: State):
            print("LLM seems to not be able to call tool. Deploying counter measure.")
            command = ""
            for line in state["last_result_content"].splitlines():
                if "**Action**:" not in line:
                    continue
                command = line.split("**Action**:", 1)[1].strip()
                print(f"Command detected in action: {command}")
                break
            else:
                print(
                    "It seems there is neither action nor tool calls. There is nothing we can do. Giving up."
                )
                return {"give_up": True}

            return {
                "tool_calls": [
                    {
                        "name": "api-use-key",
                        "args": {
                            "game": state["game"],
                            "command": command,
                            "session_key": state["key"],
                        },
                        "type": "tool_call",
                    }
                ],
                "hallucinate_count": 0,
                "hallucinate_streak": state["hallucinate_streak"] + 1,
            }

        def missing_tool_calls(state: State):
            print("LLM is hallucinating and refuse to call tools.")
            return {
                "missing_tool_calls": True,
                "hallucinate_count": state["hallucinate_count"] + 1,
            }

        def should_continue(state: State):
            if state["give_up"]:
                if state["debug"]:
                    pprint.pp("Giving up from Hallucination")
                return "END"
            elif state["hallucinate_count"] > state["hallucinate_count_threshold"]:
                return "HALLUCINATE_COUNTER_MEASURE"
            elif state["current_step"] > state["maximum_step"]:
                if state["debug"]:
                    pprint.pp(f"Reach Maximum")
                return "END"
            elif state["tool_calls"]:
                if state["debug"]:
                    pprint.pp("Action")
                return "Action"
            else:
                return "MISSING_CALL"

        async def call_tools(state: State):
            assert self.session is not None
            total_result = []
            key = ""
            for tool in state["tool_calls"]:
                try:
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

                    final_result = orjson.loads(content.text)
                    key = final_result.get("new_key", state["key"])
                    if tool["name"] == "api-get-chat-log":
                        continue
                    total_result.append(
                        {
                            "tool_name": tool["name"],
                            "arguments": tool["args"],
                            "response": final_result,
                        }
                    )
                except Exception as e:
                    total_result.append(
                        {
                            "tool_name": tool["name"],
                            "arguments": tool["args"],
                            "response": f"Error: {str(e)}",
                        }
                    )

            total_result = state["history"] + total_result
            if len(total_result) > state["history_max_length"]:
                total_result = [total_result[0]] + total_result[
                    -state["history_max_length"] :
                ]
            return {"history": total_result, "key": key}

        async def summarize(state: State):
            print("Summarizing")
            try:
                assert self.session is not None
                await self.session.call_tool(
                    name="api-get-chat-log",
                    arguments={"game": state["game"], "session_key": state["key"]},
                )

                template = """Summarize the following history of commands and responses from playing zork in Json format with the following keys:
                            1. game_completed: True if the Zork game has been successfully completed by collecting all 20 treasures and achieving victory. False if the game is still in progress or if the player has died. Only set to True when the game explicitly indicates victory/winning condition met.
                            2. current_status: Current game state description including: current location, recent game response, inventory status, immediate objectives, or any important game feedback. This helps track progress and inform the next decision.
                            3. score: Current score in the Zork game, representing the number of treasures collected so far.

                            ## Please return your response as valid JSON with this exact structure##
                            {{
                                "game_completed": false,
                                "current_status": "You are in the Living Room. You have 2 treasures: a lamp and a sword. Your score is 20 out of 350. You need to find more treasures and avoid the troll.",
                                "score": 20
                            }}
                
                            ### History ###
                            {history}
                
                            """

                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | state["llm"]
                result = await chain.ainvoke({"history": state["history"][1:]})
                json_match = re.search(
                    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", result.content, re.DOTALL
                )
                assert json_match is not None, "No JSON object found in the response"
                structured_result = json.loads(json_match.group())
                return {
                    "structured_response": structured_result,
                    "current_step": state["current_step"] - 1,
                }
            except Exception as e:
                return {
                    "structured_response": f"Error: {str(e)}",
                    "current_step": state["current_step"] - 1,
                }

        graph = StateGraph(State)
        graph.add_node("llm_call", llm_call)
        graph.add_node("environment", call_tools)
        graph.add_node("summarize", summarize)
        graph.add_node("missing_tool_calls", missing_tool_calls)
        graph.add_node("hallucinate_counter_measure", hallucinate_counter_measure)
        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges(
            "llm_call",
            should_continue,
            {
                "Action": "environment",
                "END": "summarize",
                "MISSING_CALL": "missing_tool_calls",
                "HALLUCINATE_COUNTER_MEASURE": "hallucinate_counter_measure",
            },
        )
        graph.add_edge("missing_tool_calls", "llm_call")
        graph.add_edge("hallucinate_counter_measure", "environment")
        graph.add_edge("environment", "llm_call")
        graph.add_edge("summarize", END)
        agent = graph.compile()
        self.agent = agent

    async def talk_with_zork(
        self,
        model: ChatGoogleGenerativeAI | ChatOllama,
        category: str,
        debug: bool,
    ):
        match category:
            case "standard_prompting":
                template = """ You are playing Zork (text-based game) via accessing MCP tool
               
                            Goal: collect as many treasures as possible.
                                                
                            ### Previous result ###
                                    {history}

                            ### Instruction ###
                                    - Based on the previous result, use the previous generated key and assign proper command to play Zork
                                                    
                            ## Important ##
                                    - Do not repeat the same command
                                    - Do not stop and keep playing
                            """
                agent_response = await self.agent.ainvoke(
                    {
                        "llm": model,
                        "template": template,
                        "history": [],
                        "history_max_length": 20,
                        "tool_calls": [],
                        "last_result_content": "",
                        "structured_response": None,
                        "current_step": 0,
                        "maximum_step": 1200,
                        "debug": debug,
                        "key": "",
                        "game": "zork1",
                        "missing_tool_calls": False,
                        "hallucinate_count": 0,
                        "hallucinate_count_threshold": 3,
                        "hallucinate_streak": 0,
                        "give_up": False,
                    },
                    config={"recursion_limit": 1200},
                    # Recursion limit should be > 2 * maximum step
                )

            case "react_prompting":
                template = """ You are playing Zork (text-based game) via accessing MCP tool
                
                        Goal: collect as many treasures as possible.
                        
                        ### Previous result ###
                        {history}

                        ### Instruction ###
                            - Based on the previous result, use the previous generated key and assign proper command to play zork
                            
                        ## Important ##
                            - Avoid repeating the same command multiple times in a row
                            - You must follow the ReAct Prompt, and think like ReAct Prompt
                         
                        ## ReAct Prompt ##
                            You are an expert Zork 1 player using ReAct (Reasoning + Acting) methodology. Your goal is to achieve the maximum 350 points by collecting all 20 treasures and placing them in the Living Room trophy case.

                            Follow this exact format for every turn:

                            **Thought**: [Analyze current situation, state your reasoning, plan next action based on game state, inventory, and objectives. Make sure your analysis is concise and efficient]

                            **Action**: [Single Zork 1 command]

                            **Observation**: [Game's response - wait for this before next thought]

                            KEY OBJECTIVES:
                            - Collect all 20 treasures and place in trophy case (350 total points)
                            - Manage inventory carefully (weight/space limits)
                            - Essential items: lamp (light source), sword (combat), garlic (vampire bat)
                            - The thief steals treasures but you get them back when you defeat him

                            CRITICAL REASONING PATTERNS:

                            1. EXPLORATION: "I need to systematically explore [location] because [reason]. I should check [specific elements] and look for [items/exits/clues]."

                            2. INVENTORY: "I'm carrying [list items]. I need [X] for [purpose]. I should drop [Y] because [reason] or keep [Z] because [essential for upcoming task]."

                            3. COMBAT: "Facing [enemy]. My equipment: [list]. Strategy: [specific approach] because [tactical reasoning]. Save state first."

                            4. PUZZLE-SOLVING: "This puzzle requires [analysis]. Possible solutions: [options]. I'll try [chosen method] because [logical reasoning]."

                            5. TREASURE MANAGEMENT: "Found [treasure] worth [points]. Route to trophy case: [path]. Risks: [thief/dangers]. Priority level: [high/medium/low]."

                            GAME SEQUENCE KNOWLEDGE:
                            - Above Ground: mailbox→forest→climb tree→get egg→house→kitchen supplies→living room→down trap door
                            - Underground: defeat troll→maze→cyclops (say "Odysseus")→collect treasures systematically
                            - End game: defeat thief with nasty knife→final treasures→barrow entrance

                            ADAPTIVE REASONING:
                            - If plan fails: "That didn't work because [analysis]. Alternative approach: [new strategy] because [reasoning]."
                            - If unexpected situation: "This wasn't anticipated. Based on game logic: [analysis]. New plan: [strategy]."
                            - If thief interferes: "Thief took [item]. Impact: [assessment]. Adjust plan: [modification]."

                            Start with:
                            **Thought 1**: I'm beginning Zork 1. My ultimate goal is 350 points from all treasures in the trophy case. I should start by examining my surroundings and getting the leaflet from the mailbox for initial game information.

                            **Action 1**: look
                        """

                agent_response = await self.agent.ainvoke(
                    {
                        "llm": model,
                        "template": template,
                        "history": [],
                        "history_max_length": 20,
                        "tool_calls": [],
                        "last_result_content": "",
                        "structured_response": None,
                        "current_step": 0,
                        "maximum_step": 1200,
                        "debug": debug,
                        "key": "",
                        "game": "zork1",
                        "missing_tool_calls": False,
                        "hallucinate_count": 0,
                        "hallucinate_count_threshold": 3,
                        "hallucinate_streak": 0,
                        "give_up": False,
                    },
                    config={"recursion_limit": 1200},
                    # Recursion limit should be > 2 * maximum step
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
        "hallucinate_count",
        "hallucinate_count_threshold",
        "hallucinate_streak",
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
        category = "standard_prompting"
        debug = True
        for i in tqdm(iterable=range(1)):
            result = await client.talk_with_zork(
                model=model, category=category, debug=debug
            )
            pprint.pp(f"Iteration {i+1} completed.")

            final_result = result["structured_response"]
            # final_result = result["structured_response"].model_dump()
            # for key, value in final_result.items():
            #     print(f"{key} : {value}")

            # print(f"current_step : {result['current_step']}")
            # print(f"maximum_step : {result['maximum_step']}")
            final_result["current_step"] = result["current_step"]
            final_result["maximum_step"] = result["maximum_step"]
            final_result["hallucinate_count"] = result["hallucinate_count"]
            final_result["hallucinate_count_threshold"] = result[
                "hallucinate_count_threshold"
            ]
            final_result["hallucinate_streak"] = result["hallucinate_streak"]

            test = [final_result]
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
