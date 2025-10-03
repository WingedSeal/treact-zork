import logging
import requests
from .csv_logger import CSVLogger
from requests.exceptions import Timeout, ConnectionError, HTTPError
from .ai_model_response import AIModelResponse
from .ai_mode import AIMode
from .tool_call import ToolCall, ToolCallResult, ToolServerResponse
from langchain_core.language_models import BaseChatModel
from mcp import ClientSession
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import load_mcp_tools, BaseTool
import pprint
import os
from typing import cast, Literal, Hashable
from contextlib import AsyncExitStack
from mcp.client.streamable_http import streamablehttp_client
from langchain_core.messages import (
    SystemMessage,
    AIMessageChunk,
    BaseMessageChunk,
    ToolCall as LangChainToolCall,
)
from langchain_ollama import ChatOllama
import orjson
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph
import csv
import datetime
from tqdm import tqdm
from mcp.types import TextContent
import time
from treact_client.load_env import env
from .state import State, PartialState
from . import prompt_template
from .log import get_logger

logger = get_logger(__name__)


SAVE_HISTORY = 10

if env.API_KEY:
    os.environ["GOOGLE_API_KEY"] = env.API_KEY
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.5,
        # max_output_tokens=8000,
        include_thoughts=True,
        thinking_budget=1024,
    )
else:
    model = ChatOllama(model="qwen3:8b", temperature=0)
    # model = ChatOllama(model="gpt-oss:20b", temperature=0)
    # model = ChatOllama(model="llama3.1:latest", temperature=0)

MCP_SERVER_URL = f"http://{env.SERVER_IP}:{env.SERVER_PORT}/mcp"


SHOULD_CONTINUE_TYPE = Literal[
    "END", "ACTION", "MISSING_CALL", "HALLUCINATE_COUNTER_MEASURE"
]


class MCPClient:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []
        self.agent: CompiledStateGraph | None = None

    def is_server_up(self) -> bool:
        try:
            response = requests.get(MCP_SERVER_URL, timeout=3)
        except (ConnectionError, Timeout, HTTPError):
            return False
        return response.status_code % 100 != 5

    async def _get_session(self) -> ClientSession:
        read_stream, write_stream, session_ID = (
            await self.exit_stack.enter_async_context(
                streamablehttp_client(MCP_SERVER_URL)
            )
        )
        # print(f"Streams acquired: {read_stream=}, {write_stream=}")
        return await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def connect_to_server(self) -> None:
        if not self.is_server_up():
            raise Exception(f"MCP server at {MCP_SERVER_URL} is down.")
        self.session = await self._get_session()
        await self.session.initialize()
        logger.info(f"Connected to MCP server at {MCP_SERVER_URL}")
        self.tools = await load_mcp_tools(self.session)
        logger.info(f"Tools loaded: {', '.join(tool.name for tool in self.tools)}")
        self.agent = self._get_graph().compile()

    def _get_graph(self) -> StateGraph:
        async def llm_call(state: State) -> PartialState:
            logger.info(f"Executing {llm_call.__name__}: {state['current_step']}")
            if state["current_step"] == 0:
                logger.debug("Bypassing LLM. Calling 'gen-key'.")
                return {
                    "tool_calls": [
                        ToolCall(tool_name="gen-key", arguments={"game": state["game"]})
                    ],
                    "last_result_content": "",
                    "current_step": state["current_step"] + 1,
                }

            llm = state["llm"]
            llm_with_tools = llm.bind_tools(self.tools)

            template = state["template"]
            prompt = ChatPromptTemplate.from_template(template)

            if state["missing_tool_calls"]:
                logger.warning("Missing tool call. Reprompting")
                prompt.append(
                    SystemMessage(
                        content="You are not calling any tools, and the game has not yet ended. You put your command in **Actions** but forgot to also put it in tool calls. Call some tools."
                    )
                )

            chain = prompt | llm_with_tools

            ai_message_result: BaseMessageChunk = AIMessageChunk(content="")
            logger.info("Thinking")
            if env.API_KEY:
                logger.debug("API_KEY detected. Sleeping to prevent API rate limit.")
                time.sleep(8)  # Prevent Gemini from exploding

            logger.debug("Thoughts:")
            async for chunk in chain.astream(
                {
                    "history": state["history"],
                    "last_result_content": state["last_result_content"],
                }
            ):
                ai_tool_calls: list[LangChainToolCall] | None = getattr(
                    chunk, "tool_calls", None
                )
                if logger.should_log_to_stdio(logging.DEBUG):
                    if ai_tool_calls is not None:
                        print(f"\n[Tool Call]: {ai_tool_calls}\n", flush=True)
                    elif getattr(chunk, "content", None):
                        print(chunk.content, end="", flush=True)
                ai_message_result += chunk
            logger.info("Done thinking")

            langchain_tool_calls: list[LangChainToolCall] = getattr(
                ai_message_result, "tool_calls", []
            )
            # logger.info(f"LLM Response Content: {ai_message_result.content}")
            # logger.info(f"LLM Tool Calls: {langchain_tool_calls}")

            tool_calls = [
                ToolCall(tool_name=call["name"], arguments=call["args"])
                for call in langchain_tool_calls
            ]

            return {
                "missing_tool_calls": False,
                "tool_calls": tool_calls,
                "last_result_content": ai_message_result.content,
                "current_step": state["current_step"] + 1,
            }

        def missing_tool_calls(state: State) -> PartialState:
            logger.warning("LLM did not call any tool.")
            return {
                "missing_tool_calls": True,
                "hallucinate_count": state["hallucinate_count"] + 1,
            }

        def should_continue(state: State) -> SHOULD_CONTINUE_TYPE:
            if state["hallucinate_count"] > state["hallucinate_count_threshold"]:
                logger.warning("Reaching hallucination threshold. Ending the game.")
                return "END"
            elif state["current_step"] > state["maximum_step"]:
                logger.info(
                    f"Reaching maximum step ({state['maximum_step']}). Ending the game."
                )
                return "END"
            elif state["tool_calls"]:
                return "ACTION"
            else:
                return "MISSING_CALL"

        async def call_tools(state: State) -> PartialState:
            logger.info(f"Executing {call_tools.__name__}: {state['current_step']}")
            assert self.session is not None
            tool_call_results: list[ToolCallResult] = []
            key = ""
            for tool in state["tool_calls"]:
                logger.debug(f"Calling {tool.tool_name}.")
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
                    logger.error(e)
                    continue

                content = result.content[0]
                if not isinstance(content, TextContent):
                    raise Exception(
                        f"Expected TextContent from tool (got {type(content)})."
                    )
                server_response: ToolServerResponse = orjson.loads(content.text)
                logger.info(f"Tool Server Response: {server_response}.")
                key = server_response.get("new_key", state["key"])
                if tool.tool_name == "get-chat-log":
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

        async def summarize(state: State) -> PartialState:
            logger.info(f"Executing {call_tools.__name__}: {state['current_step']}")
            assert self.session is not None
            logger.debug(f"Calling get-chat-log.")
            try:
                await self.session.call_tool(
                    name="get-chat-log",
                    arguments={"game": state["game"], "session_key": state["key"]},
                )
            except Exception as e:
                logger.error(e)
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
            structured_response: AIModelResponse = cast(
                AIModelResponse, await chain.ainvoke({"history": state["history"]})
            )
            logger.info(f"Structured Response: {structured_response}")
            return {
                "structured_response": structured_response,
                "current_step": state["current_step"] - 1,
            }

        graph = StateGraph(State)
        graph.add_node("llm_call", llm_call)
        graph.add_node("environment", call_tools)
        graph.add_node("summarize", summarize)
        graph.add_node("missing_tool_calls", missing_tool_calls)
        graph.add_edge(START, "llm_call")
        path_map: dict[SHOULD_CONTINUE_TYPE, str] = {
            "ACTION": "environment",
            "END": "summarize",
            "MISSING_CALL": "missing_tool_calls",
        }

        graph.add_conditional_edges(
            "llm_call", should_continue, cast(dict[Hashable, str], path_map)
        )
        graph.add_edge("missing_tool_calls", "llm_call")
        graph.add_edge("environment", "llm_call")
        graph.add_edge("summarize", END)
        return graph

    async def talk_with_zork(
        self,
        model: BaseChatModel,
        ai_mode: AIMode,
    ) -> State:
        assert self.agent is not None
        match ai_mode:
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
                            "maximum_step": 200,
                            "key": "",
                            "game": "zork1",
                            "missing_tool_calls": False,
                            "hallucinate_count": 0,
                            "hallucinate_count_threshold": 5,
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
                            "key": "",
                            "game": "zork1",
                            "missing_tool_calls": False,
                            "hallucinate_count": 0,
                            "hallucinate_count_threshold": 3,
                        }
                    ),
                    config={"recursion_limit": 1200},
                    # Recursion limit should be > 2 * maximum step
                )
            case AIMode.TREACT:
                raise NotImplementedError()

        return cast(State, agent_response)

    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.exit_stack.aclose()


async def run_client(ai_mode: AIMode, iterations: int = 3) -> None:
    logging.info(f"Running MCP-Client as {ai_mode.value} mode")
    client = MCPClient()
    await client.connect_to_server()
    csv = CSVLogger(model, ai_mode)
    for i in tqdm(iterable=range(iterations)):
        logging.info(f"Talking with Zork. Iteration: {i+1}/{iterations}")
        result_state = cast(
            State,
            await client.talk_with_zork(model=model, ai_mode=ai_mode),
        )
        csv.add_result_state(result_state)

    await client.cleanup()
