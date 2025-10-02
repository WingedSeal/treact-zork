from .ai_model_response import AIModelResponse
from .ai_mode import AIMode
from .tool_call import ToolCall, ToolCallResult, ToolServerResponse
from langchain_core.language_models import BaseChatModel
from mcp import ClientSession
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools, BaseTool
import pprint
import os
from typing import Optional, Any
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.messages import SystemMessage, AIMessageChunk, AIMessage
from langchain_ollama import ChatOllama
from typing import Any
import orjson
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph
import csv
import datetime
from tqdm import tqdm
from mcp.types import TextContent
import time
from treact_client.load_env import env
from pathlib import Path
from .state import State
from . import prompt_template
from .logger import setup_logger, LOG_DIRECTORY

logger = setup_logger()


SAVE_HISTORY = 10

if env.API_KEY:
    os.environ["GOOGLE_API_KEY"] = env.API_KEY
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        # max_output_tokens=8000,
        include_thoughts=True,
        thinking_budget=1024,
    )
else:
    model = ChatOllama(model="qwen3:8b", temperature=0)
    # model = ChatOllama(model="gpt-oss:20b", temperature=0)
    # model = ChatOllama(model="llama3.1:latest", temperature=0)


class MCPClient:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []
        self.agent: CompiledStateGraph | None = None

    async def _get_session(self) -> ClientSession:
        read_stream, write_stream, session_ID = (
            await self.exit_stack.enter_async_context(
                streamablehttp_client(f"http://{env.SERVER_IP}:{env.SERVER_PORT}/mcp")
            )
        )
        print(f"Streams acquired: {read_stream=}, {write_stream=}")
        return await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def connect_to_server(self) -> None:
        self.session = await self._get_session()
        await self.session.initialize()
        self.tools = await load_mcp_tools(self.session)
        pprint.pp(self.tools)
        print("Connected to server with tools:", [tool.name for tool in self.tools])
        self.agent = self._get_graph().compile()

    def _get_graph(self) -> StateGraph:
        async def llm_call(state: State):
            if state["current_step"] == 0:
                return {
                    "tool_calls": [
                        ToolCall(
                            tool_name="api-gen-key", arguments={"game": state["game"]}
                        )
                    ],
                    "last_result_content": "",
                    "current_step": state["current_step"] + 1,
                }

            llm = state["llm"]
            llm_with_tools = llm.bind_tools(self.tools)

            template = state["template"]
            prompt = ChatPromptTemplate.from_template(template)

            if state["missing_tool_calls"]:
                state["missing_tool_calls"] = False
                prompt.append(
                    SystemMessage(
                        content="You are not calling any tools, and the game has not yet ended. You put your command in **Actions** but forgot to also put it in tool calls. Call some tools."
                    )
                )

            chain = prompt | llm_with_tools

            ai_message_result = AIMessageChunk(content="")
            print("-- THINKING TIME --\n")
            if env.API_KEY:
                time.sleep(5)  # Prevent Gemini from exploding

            async for chunk in chain.astream(
                {
                    "history": state["history"],
                    "current_step": state["current_step"],
                    "maximum_step": state["maximum_step"],
                }
            ):
                ai_tool_calls = getattr(chunk, "tool_calls", None)
                if ai_tool_calls is not None:
                    print(f"\n[Tool Call]: {ai_tool_calls}", flush=True)
                elif getattr(chunk, "content", None):
                    print(chunk.content, end="", flush=True)
                ai_message_result += chunk
            print("\n-- DONE THINKING --\n")

            if state["debug"]:
                print(f"Step = {state['current_step']}")
                print("\nFull LLM Response:")
                print(f"Content: {ai_message_result.content}")
                print(
                    f"Response Metadata: {getattr(ai_message_result, 'response_metadata', {})}"
                )
                print(f"Tool Calls: {getattr(ai_message_result, 'tool_calls', [])}")
                print(f"type: {type(ai_message_result)}")

            tool_calls = [
                ToolCall(tool_name=call["name"], arguments=call["args"])
                for call in getattr(ai_message_result, "tool_calls", [])
            ]
            print(tool_calls)
            last_result_content = ai_message_result.content
            return {
                "tool_calls": tool_calls,
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
                    ToolCall(
                        tool_name="api-use-key",
                        arguments={
                            "game": state["game"],
                            "command": command,
                            "session_key": state["key"],
                        },
                    )
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
            tool_call_results: list[ToolCallResult] = []
            key = ""
            for tool in state["tool_calls"]:
                try:
                    result = await self.session.call_tool(
                        tool.tool_name, arguments=tool.arguments
                    )
                except Exception as e:
                    tool_call_results.append(
                        ToolCallResult(
                            tool_name=tool.tool_name,
                            arguments=tool.arguments,
                            tool_server_response={"MCP-Server Error": str(e)},
                        )
                    )
                    continue

                if state["debug"]:
                    pprint.pp(result)
                content = result.content[0]
                if not isinstance(content, TextContent):
                    raise Exception(
                        f"It is not TextContent since it is {type(content)}"
                    )
                if state["debug"]:
                    pprint.pp(content)

                server_response: ToolServerResponse = orjson.loads(content.text)
                key = server_response.get("new_key", state["key"])
                if tool.tool_name == "api-get-chat-log":
                    continue
                tool_call_results.append(
                    ToolCallResult(
                        tool_name=tool.tool_name,
                        arguments=tool.arguments,
                        tool_server_response=server_response,
                    )
                )
            tool_call_results = state["history"] + tool_call_results
            if len(tool_call_results) > state["history_max_length"]:
                tool_call_results = tool_call_results[-state["history_max_length"] :]
            return {"history": tool_call_results, "key": key}

        async def summarize(state: State):
            print("Summarizing")
            assert self.session is not None
            try:
                await self.session.call_tool(
                    name="api-get-chat-log",
                    arguments={"game": state["game"], "session_key": state["key"]},
                )
            except Exception as e:
                return {
                    "structured_response": AIModelResponse(
                        game_completed=False, current_status=str(e), score=0
                    ),
                    "current_step": state["current_step"] - 1,
                }

            structured_llm = state["llm"].with_structured_output(AIModelResponse)

            template = prompt_template.SUMMARIZE

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | structured_llm
            result = await chain.ainvoke(
                {"history": state["history"][-SAVE_HISTORY:-1]}
            )
            return {
                "structured_response": result,
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
        return graph

    async def talk_with_zork(
        self,
        model: BaseChatModel,
        category: AIMode,
        debug: bool,
    ):
        assert self.agent is not None
        match category:
            case AIMode.STANDARD:
                agent_response = await self.agent.ainvoke(
                    State(
                        {
                            "llm": model,
                            "template": prompt_template.STANDARD,
                            "history": [],
                            "history_max_length": 10,
                            "tool_calls": [],
                            "last_result_content": "",
                            "structured_response": None,
                            "current_step": 0,
                            "maximum_step": 400,
                            "debug": debug,
                            "key": "",
                            "game": "zork1",
                            "missing_tool_calls": False,
                            "hallucinate_count": 0,
                            "hallucinate_count_threshold": 3,
                            "hallucinate_streak": 0,
                            "give_up": False,
                        }
                    ),
                    config={"recursion_limit": 1200},
                    # Recursion limit should be > 2 * maximum step
                )

            case AIMode.REACT:

                agent_response = await self.agent.ainvoke(
                    State(
                        {
                            "llm": model,
                            "template": prompt_template.REACT,
                            "history": [],
                            "history_max_length": 40,
                            "tool_calls": [],
                            "last_result_content": "",
                            "structured_response": None,
                            "current_step": 0,
                            "maximum_step": 200,
                            "debug": debug,
                            "key": "",
                            "game": "zork1",
                            "missing_tool_calls": False,
                            "hallucinate_count": 0,
                            "hallucinate_count_threshold": 3,
                            "hallucinate_streak": 0,
                            "give_up": False,
                        }
                    ),
                    config={"recursion_limit": 1200},
                    # Recursion limit should be > 2 * maximum step
                )
            case AIMode.TREACT:
                raise NotImplementedError()
            case _:
                raise Exception("Please assign the type of prompting")

        return agent_response

    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.exit_stack.aclose()


# Global client instance
client: MCPClient = MCPClient()


async def run_client():
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
    category = AIMode.STANDARD
    with open(
        f"{LOG_DIRECTORY}/{category}_{model.model.replace('/','-').replace(':', '-')}_{current}.csv",
        "w",
        newline="",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
    print(f"Model: {model.model}")
    try:

        debug = True
        for i in tqdm(iterable=range(5)):
            result = await client.talk_with_zork(
                model=model, category=category, debug=debug
            )
            pprint.pp(f"Iteration {i+1} completed.")

            # final_result = result["structured_response"]
            final_result = result["structured_response"].model_dump()
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
                f"{LOG_DIRECTORY}/result_{model.model.replace('/','-').replace(':', '-')}_{current}.csv",
                "a",
                newline="",
            ) as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields)
                writer.writerows(test)
    except Exception as e:
        print(f"Error: {str(e)}")

    await client.cleanup()
