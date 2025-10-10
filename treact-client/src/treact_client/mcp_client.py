import asyncio
import logging
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path
from pprint import pformat
from typing import Hashable, Literal, Sequence, cast

import orjson
import requests
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
    HumanMessage,
    MessageLikeRepresentation,
    SystemMessage,
)
from langchain_core.messages import ToolCall as LangChainToolCall
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.tools import BaseTool, load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.graph.state import (
    END,
    START,
    CompiledStateGraph,
    RunnableConfig,
    StateGraph,
)
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tqdm import tqdm

from treact_client.client_toml import parse_toml_config

from . import prompt_template
from .ai_model_response import AIModelResponse
from .csv_logger import CSVLogger
from .exceptions import (
    InvalidToolCallResultException,
    MaxBranchPerNodeExceededException,
    NoSessionException,
    UnreachableServerException,
)
from .load_env import env
from .log import get_logger
from .state import ModelSettings, State, StateUpdate
from .tool_call import (
    ToolCall,
    ToolCallResult,
    ToolCallResultNode,
    ToolCallResultNodeUpdate,
    ToolServerResponse,
)

logger = get_logger(__name__)


MCP_SERVER_URL = f"http://{env.SERVER_IP}:{env.SERVER_PORT}/mcp"


ACTIONS_AFTER_LLM_CALL = Literal[
    "END_AND_SUMMARIZE",
    "CALL_TOOLS",
    "HANDLE_LEN_TOOL_CALLS_OUT_OF_RANGE",
]


class MCPClient:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[BaseTool] = []
        self.agent: CompiledStateGraph | None = None

    def is_server_up(self, url: str) -> bool:
        try:
            response = requests.get(url, timeout=5)
        except (ConnectionError, Timeout, HTTPError):
            return False
        return response.status_code % 100 != 5

    async def _get_session(self, url: str) -> ClientSession:
        read_stream, write_stream, session_ID = (
            await self.exit_stack.enter_async_context(streamablehttp_client(url))
        )
        logger.debug(f"Session Acquired: {session_ID}")
        # print(f"Streams acquired: {read_stream=}, {write_stream=}")
        return await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

    async def connect_to_server(self, url: str) -> None:
        if not self.is_server_up(url):
            raise UnreachableServerException(url)
        self.session = await self._get_session(url)
        await self.session.initialize()
        logger.info(f"Connected to MCP server at {url}")
        self.tools = await load_mcp_tools(self.session)
        logger.info(f"Tools loaded: {', '.join(tool.name for tool in self.tools)}")
        self.agent = self._get_graph().compile()

    def _get_graph(self) -> StateGraph:
        async def call_llm(state: State) -> StateUpdate:
            logger.info(f"Executing {call_llm.__name__}: {state['current_step']}")
            if state["current_step"] == 0:
                logger.debug("Bypassing LLM. Calling 'gen-key'.")
                return {
                    "tool_calls": [
                        ToolCall(
                            tool_name="gen-key",
                            arguments={"game_name": state["model_settings"].game_name},
                            ai_thought="Let's start with gen-key",
                        )
                    ],
                    "current_step": state["current_step"] + 1,
                }

            llm = state["model_settings"].llm
            llm_with_tools = llm.bind_tools(self.tools)

            template = state["model_settings"].prompt_template
            # prompt = ChatPromptTemplate.from_template(template)

            tool_call_result_node: ToolCallResultNode = state[
                "tool_call_result_history_tree"
            ].peek()
            messages: Sequence[MessageLikeRepresentation] = (
                [SystemMessage(content=template)]
                + [
                    msg
                    for tool_call_result in tool_call_result_node.get_history(
                        state["model_settings"].history_max_length
                    )
                    for msg in (
                        AIMessage(content=tool_call_result.ai_thought or ""),
                        HumanMessage(content=str(tool_call_result)),
                    )
                ]
                + [
                    msg
                    for len_call_tools, ai_thought_len_tool_calls_out_of_range in state[
                        "ai_thoughts_len_tool_calls_out_of_range"
                    ]
                    for msg in (
                        AIMessage(content=ai_thought_len_tool_calls_out_of_range),
                        SystemMessage(
                            content=f"You are expected to call between "
                            f"{state['model_settings'].min_tool_calls} and "
                            f"{state['model_settings'].max_tool_calls}. "
                            f"But you called {len_call_tools} tool(s)."
                        ),
                    )
                ]
            )
            prompt = ChatPromptTemplate.from_messages(messages)

            if state["ai_thoughts_len_tool_calls_out_of_range"]:
                logger.warning(
                    f"Length of tool calls is out of range. Reprompting ({len(state['ai_thoughts_len_tool_calls_out_of_range'])})"
                )

            chain = prompt | llm_with_tools

            ai_message_result: BaseMessageChunk = AIMessageChunk(content="")
            logger.info("Thinking")
            if env.API_KEY:
                logger.debug("API_KEY detected. Sleeping to prevent API rate limit.")
                time.sleep(8)  # Prevent Gemini from exploding

            logger.debug("Thoughts:")
            async for chunk in chain.astream({}):
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

            logger.debug(f"LLM Response Content: {ai_message_result.content}")
            logger.debug(f"LLM Tool Calls: {langchain_tool_calls}")

            tool_calls = [
                ToolCall(
                    tool_name=call["name"],
                    arguments=call["args"],
                    ai_thought=ai_message_result.content,
                )
                for call in langchain_tool_calls
            ]

            return {
                "tool_calls": tool_calls,
                "tool_calls_parent": tool_call_result_node,
                "current_step": state["current_step"] + 1,
                "last_ai_thought": ai_message_result.content,
            }

        def handle_len_tool_calls_out_of_range(state: State) -> StateUpdate:
            logger.warning(
                f"LLM is expected to call between {state['model_settings'].min_tool_calls} "
                f"and {state['model_settings'].max_tool_calls} tool(s) but it called "
                f"{len(state['tool_calls'])} tool(s)."
            )
            assert state["last_ai_thought"] is not None
            return {
                "len_tool_calls_out_of_range_count": state[
                    "len_tool_calls_out_of_range_count"
                ]
                + 1,
                "ai_thoughts_len_tool_calls_out_of_range": state[
                    "ai_thoughts_len_tool_calls_out_of_range"
                ]
                + [(len(state["tool_calls"]), state["last_ai_thought"])],
            }

        def should_continue(state: State) -> ACTIONS_AFTER_LLM_CALL:
            if (
                len(state["ai_thoughts_len_tool_calls_out_of_range"])
                >= state["len_tool_calls_out_of_range_threshold"]
            ):
                logger.warning("Reaching hallucination threshold. Ending the game.")
                return "END_AND_SUMMARIZE"
            elif state["current_step"] > state["maximum_step"]:
                logger.info(
                    f"Reaching maximum step ({state['maximum_step']}). Ending the game."
                )
                return "END_AND_SUMMARIZE"
            elif (
                state["model_settings"].min_tool_calls
                <= len(state["tool_calls"])
                <= state["model_settings"].max_tool_calls
            ):
                return "CALL_TOOLS"
            else:
                return "HANDLE_LEN_TOOL_CALLS_OUT_OF_RANGE"

        async def call_tools(state: State) -> StateUpdate:
            logger.info(f"Executing {call_tools.__name__}: {state['current_step']}")
            tool_call_results: list[ToolCallResult] = []
            for tool in state["tool_calls"]:
                tool_call_results.append(await self.call_tool(tool))

            return {
                "tool_call_result_history_tree": ToolCallResultNodeUpdate.Pop(),
                "tool_call_results": tool_call_results,
                "ai_thoughts_len_tool_calls_out_of_range": [],
            }

        async def prune_tool_call_results(state: State) -> StateUpdate:
            tool_call_results = state["tool_call_results"]

            # TODO: evaluate good nodes

            if len(tool_call_results) > state["model_settings"].max_branch_per_node:
                raise MaxBranchPerNodeExceededException(
                    len(tool_call_results), state["model_settings"].max_branch_per_node
                )
            return {
                "tool_call_result_history_tree": ToolCallResultNodeUpdate.PutBack(
                    [
                        ToolCallResultNode(tool_call_result, state["tool_calls_parent"])
                        for tool_call_result in tool_call_results
                    ]
                ),
            }

        async def end_and_summarize(state: State) -> StateUpdate:
            logger.info(f"Executing {call_tools.__name__}: {state['current_step']}")
            if self.session is None:
                raise NoSessionException()

            tool_call_result_nodes: list[ToolCallResultNode] = list(
                state["tool_call_result_history_tree"].queue
            )

            logger.info(
                f"{len(tool_call_result_nodes)} leaf node(s) are left in the history tree."
            )
            logger.debug(f"ToolCallResultNodes are {pformat(tool_call_result_nodes)}")

            chosen_tool_call_result_node = tool_call_result_nodes[
                0
            ]  # TODO: Evaluate the best node

            logger.info(f"The chosen node is {chosen_tool_call_result_node}")

            logger.debug(f"Sending 'inventory' command")
            tool_call_result_with_inventory = await self.call_tool(
                ToolCall(
                    tool_name="use-key",
                    arguments={
                        "game_name": state["model_settings"].game_name,
                        "command": "inventory",
                        "session_key": chosen_tool_call_result_node.tool_call_result.tool_server_response[
                            "new_key"
                        ],
                    },
                )
            )

            tool_call_result_node_with_inventory = ToolCallResultNode(
                tool_call_result_with_inventory, chosen_tool_call_result_node
            )

            logger.debug("Calling get-chat-log")
            chat_log = await self.call_tool(
                ToolCall(
                    tool_name="get-chat-log",
                    arguments={
                        "game_name": state["model_settings"].game_name,
                        "session_key": tool_call_result_with_inventory.tool_server_response[
                            "new_key"
                        ],
                    },
                )
            )
            logger.info(f"Chat Log: \n {chat_log}")

            llm_with_structured_output = state[
                "model_settings"
            ].llm.with_structured_output(AIModelResponse)

            prompt = PromptTemplate.from_template(prompt_template.SUMMARIZE)

            # prompt = ChatPromptTemplate.from_messages(
            #     [
            #         ("system", template),
            #     ]
            # )
            chain = prompt | llm_with_structured_output
            ai_model_response = cast(
                AIModelResponse,
                await chain.ainvoke(
                    {
                        "tool_call_result_history": tool_call_result_node_with_inventory.get_history(
                            state["model_settings"].history_max_length
                        )
                    }
                ),
            )
            logger.info(f"Structured Response: {ai_model_response}")
            return {
                "ai_model_response": ai_model_response,
            }

        graph = StateGraph(State)
        graph.add_node("call_llm", call_llm)
        graph.add_node("call_tools", call_tools)
        graph.add_node("prune_tool_call_results", prune_tool_call_results)
        graph.add_node("end_and_summarize", end_and_summarize)
        graph.add_node(
            "handle_len_tool_calls_out_of_range", handle_len_tool_calls_out_of_range
        )

        graph.add_edge(START, "call_llm")
        path_map: dict[ACTIONS_AFTER_LLM_CALL, str] = {
            "CALL_TOOLS": "call_tools",
            "END_AND_SUMMARIZE": "end_and_summarize",
            "HANDLE_LEN_TOOL_CALLS_OUT_OF_RANGE": "handle_len_tool_calls_out_of_range",
        }
        graph.add_conditional_edges(
            "call_llm", should_continue, cast(dict[Hashable, str], path_map)
        )
        graph.add_edge("handle_len_tool_calls_out_of_range", "call_llm")
        graph.add_edge("call_tools", "prune_tool_call_results")
        graph.add_edge("prune_tool_call_results", "call_llm")
        graph.add_edge("end_and_summarize", END)
        return graph

    async def call_tool(self, tool: ToolCall) -> ToolCallResult:
        if self.session is None:
            raise NoSessionException()
        logger.debug(f"Calling {tool.tool_name}.")
        call_tool_result = await self.session.call_tool(
            tool.tool_name, arguments=tool.arguments
        )

        content = call_tool_result.content[0]
        if not isinstance(content, TextContent):
            raise InvalidToolCallResultException(type(content))
        server_response: ToolServerResponse = orjson.loads(content.text)
        logger.info(f"Tool Server Response: {server_response}.")
        return ToolCallResult(
            tool_name=tool.tool_name,
            arguments=tool.arguments,
            ai_thought=tool.ai_thought,
            tool_server_response=server_response,
        )

    async def invoke_agent(
        self, model_settings: ModelSettings, config: RunnableConfig
    ) -> State:
        """
        Recursion limit should be > 2 * maximum step
        """
        if self.agent is None:
            raise NoSessionException()
        return cast(
            State, await self.agent.ainvoke(model_settings.to_new_state(), config)
        )

    async def cleanup(self) -> None:
        """Clean up resources"""
        await self.exit_stack.aclose()


async def run_client(
    client_name: str,
    *,
    prompt_template: str,
    game_name: str,
    maximum_step: int,
    len_tool_calls_out_of_range_threshold: int,
    history_max_length: int,
    max_branch_per_node: int,
    min_tool_calls: int,
    max_tool_calls: int,
    iterations: int,
    config: RunnableConfig,
) -> None:
    logging.info(f"Running MCP-Client: {client_name}")
    if env.API_KEY:
        logging.info(f"API_KEY found. Using Gemini.")
        os.environ["GOOGLE_API_KEY"] = env.API_KEY
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.6,
            # max_output_tokens=8000,
            include_thoughts=True,
            thinking_budget=1024,
        )
    else:
        logging.info(f"API_KEY not found. Using Ollama.")
        model = ChatOllama(model="qwen3:8b", temperature=0)
        # model = ChatOllama(model="gpt-oss:20b", temperature=0)
        # model = ChatOllama(model="llama3.1:latest", temperature=0)
    client = MCPClient()
    await client.connect_to_server(MCP_SERVER_URL)
    csv = CSVLogger(model, client_name)
    for i in tqdm(iterable=range(iterations)):
        logging.info(f"Talking with Zork. Iteration: {i+1}/{iterations}")
        result_state = await client.invoke_agent(
            ModelSettings(
                llm=model,
                prompt_template=prompt_template,
                game_name=game_name,
                maximum_step=maximum_step,
                len_tool_calls_out_of_range_threshold=len_tool_calls_out_of_range_threshold,
                min_tool_calls=min_tool_calls,
                max_tool_calls=max_tool_calls,
                history_max_length=history_max_length,
                max_branch_per_node=max_branch_per_node,
            ),
            config=config,
        )
        csv.add_result_state(result_state)

    await client.cleanup()


def run_client_file(toml_path: Path) -> None:
    asyncio.run(run_client(**parse_toml_config(toml_path)))
