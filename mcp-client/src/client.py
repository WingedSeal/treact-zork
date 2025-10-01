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
        include_thoughts=True,
        thinking_budget=1024,
    )
else:
    # model = ChatOllama(model="gpt-oss:20b", temperature=0)
    model = ChatOllama(model="qwen3:8b", temperature=0)
    # model = ChatOllama(model="llama3.1:latest", temperature=0)


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
            # if state["current_step"] == 0:
            #     final_result = [
            #         {
            #             "name": "api-get-dict",
            #             "args": {"game": state["game"]},
            #             "type": "tool_call",
            #         }
            #     ]

            if state["current_step"] == 0:
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
                time.sleep(2)
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
                # total_result = [total_result[0]] + total_result[
                #     -state["history_max_length"] :
                # ]
                total_result = total_result[-state["history_max_length"] :]
            return {"history": total_result, "key": key}

        async def summarize(state: State):
            print("Summarizing")
            try:
                assert self.session is not None
                await self.session.call_tool(
                    name="api-get-chat-log",
                    arguments={"game": state["game"], "session_key": state["key"]},
                )
                structured_llm = state["llm"].with_structured_output(response)

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
                chain = prompt | structured_llm
                result = await chain.ainvoke({"history": state["history"][-10:-1]})
                # json_match = re.search(
                #     r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", result.content, re.DOTALL
                # )
                # assert json_match is not None, "No JSON object found in the response"
                # structured_result = json.loads(json_match.group())
                return {
                    "structured_response": result,
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
               
                            Goal: collect as many treasures as possible and putting them in the trophy case in the Living Room of the house.
                                The following is the list of treasures and their locations:
                                (1)Above Ground
                                    - Jeweled egg
                                (2) Cellar and Maze
                                    - Bag of coins
                                    - Painting

                                (3) Dam and Reservoir
                                    - Platinum bar
                                    - Trunk of jewels
                                    - Crystal trident

                                (4) Temple Area
                                    - Ivory torch
                                    - Gold coffin
                                    - Sceptre
                                    - Crystal skull

                                (5) Old Man River
                                    - Emerald
                                    - Scarab
                                    - Pot of gold

                                (6) Coal Mine
                                    - Jade figurine
                                    - Sapphire bracelet
                                    - Diamond

                                (7) Treasure Room and Barrow
                                    - Silver chalice
                                    - Clockwork canary
                                    - Brass bauble

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
                    },
                    config={"recursion_limit": 1200},
                    # Recursion limit should be > 2 * maximum step
                )

            case "react_prompting":
                template = """ You are playing Zork (text-based game) via accessing MCP tool
                
                        Goal: collect as many treasures as possible and putting them in the trophy case in the Living Room of the house.
                                The following is the list of treasures and their locations:
                                (1)Above Ground
                                    - Jeweled egg
                                (2) Cellar and Maze
                                    - Bag of coins
                                    - Painting

                                (3) Dam and Reservoir
                                    - Platinum bar
                                    - Trunk of jewels
                                    - Crystal trident

                                (4) Temple Area
                                    - Ivory torch
                                    - Gold coffin
                                    - Sceptre
                                    - Crystal skull

                                (5) Old Man River
                                    - Emerald
                                    - Scarab
                                    - Pot of gold

                                (6) Coal Mine
                                    - Jade figurine
                                    - Sapphire bracelet
                                    - Diamond

                                (7) Treasure Room and Barrow
                                    - Silver chalice: 
                                    - Clockwork canary
                                    - Brass bauble
                        
                        ### Previous result ###
                        {history}

                        ### Instruction ###
                            - Based on the previous result, use the previous generated key and assign proper command to play zork
                            
                        ## Important ##
                            - Avoid repeating the same command multiple times in a row
                            - You must follow the ReAct Prompt, and think like ReAct Prompt
                         
                        ###Example Trajectory:###
                        
                        (1) Above Ground and Initial Descent
                            Thought: Initializing game and gathering essential items above ground. Must acquire the egg for the Thief to open later, then grab light, weapon, and container before descending.
                            Action: open mailbox
                            Observation: Mailbox opened. Get the leaflet and read it, then drop it.

                            Thought: Leaflet is no longer needed. The next steps are to find the egg and then enter the house for the lantern and sword. The egg is found in a nest North of the starting location.
                            Action: go north
                            Observation: North of House. Go north to a Forest Path location.

                            Thought: Now at the Forest Path. Climb the tree to get the egg and then descend, ensuring I attempt to open it so the Thief will process it later.
                            Action: climb tree
                            Observation: Get the egg from the nest. Go back down.

                            Thought: Must 'try to open egg' to trigger the later event with the Thief. Now, head to the house (South then East) to enter via the window.
                            Action: try to open egg
                            Observation: You find that you can't open the egg. Go south and east to Behind House.

                            Thought: At Behind House, open the window and enter the Kitchen. In the Kitchen, grab the sack and bottle, noting the garlic and potential use of the lunch/water.
                            Action: open window
                            Observation: You open the window. Enter house. Kitchen. Get sack and bottle. Open sack (contains lunch and garlic).

                            Thought: Keep the garlic and sack, as advised. Move west to the Living Room to find the main gear: the sword, the lantern, and the trap door to the dungeon.
                            Action: go west
                            Observation: Living Room. Get sword and lantern. Move rug. Open trap door.

                            Thought: Lights on and trap door open. Time to descend and begin the underground adventure. I will also save before dangerous actions, as advised.
                            Action: turn on lamp
                            Observation: The lamp is now on. Go down into the Great Underground Empire.  
                            
                        (2) Troll Encounter and Maze Navigation 
                            Thought: Now in the Cellar. The priority is to drop the egg so the Thief can open it and prepare for the battle with the troll, which is North. I should save before this risky fight.
                            Action: drop egg
                            Observation: The egg is dropped. Go north. The troll appears.

                            Thought: The troll is blocking the path. The solution guide states to attack repeatedly with the sword. I will save/restore as needed to survive this encounter.
                            Action: attack troll with sword
                            Observation: You succeed in killing the troll. Drop the sword to manage inventory limits and save again.

                            Thought: Troll defeated and path open. Now entering the Maze. The path is West, West, Up to get the key and coins. I must check my health condition after the fight.
                            Action: go west
                            Observation: Enter the Maze. Go west, west, and up. Save game.

                            Thought: I've reached the key/coins location. Get both, checking for wounds. Then the path is Southwest, East, South, and Southeast to the Cyclops Room.
                            Action: get bag of coins and skeleton key
                            Observation: Items obtained (or I must wait for healing if wounded). Go southwest, east, south, and southeast to the Cyclops Room.

                            Thought: Now at the Cyclops Room. The preferred solution is to say "Odysseus" to scare him away, which opens a direct route back to the house.
                            Action: say "Odysseus"
                            Observation: The Cyclops is scared away, destroying the wall and opening a path.

                            Thought: Return to the Living Room via the new path (East twice). Once there, deposit the first treasure (coins) in the trophy case and drop non-essential items.
                            Action: go east
                            Observation: Go east again. Living Room. Open trophy case. Put coins in case. Drop key and bottle.
                        
                        (3) Dam and Reservoir Treasures
                            Thought: Next target is the Platinum Bar from the Loud Room. The easiest method is to simply type 'echo' there after navigating: Down, North, East 3 times.
                            Action: go down
                            Observation: Go north, east 3 times. Loud Room.

                            Thought: I am in the Loud Room. Type 'echo' to trigger the bar's appearance. Then retrieve it and return to the Living Room (West 3 times, South, Up) to deposit the treasure.
                            Action: echo
                            Observation: The platinum bar appears. Get the platinum bar. Go west 3 times, south, and up. Living Room. Put platinum bar in trophy case.

                            Thought: Now to empty the Reservoir for the Trunk of Jewels. Path: Down, North, East 3 times, Up, East, North to Dam Lobby. Grab the matchbook.
                            Action: go down
                            Observation: Go north, east 3 times, up, east, and north. Dam Lobby. Get matchbook.

                            Thought: Move to the Maintenance Room to push the yellow button and get tools. Then back to the Dam to use the wrench to turn the bolt.
                            Action: go north
                            Observation: Maintenance Room. Get wrench and screwdriver. Push yellow button. Go south twice to the Dam. Turn bolt with wrench. Drop wrench.

                            Thought: Dam is open. Move to Reservoir South, wait two turns for it to empty, then go North to get the Trunk. Return path: South, Southwest, Southwest, West, South, Up.
                            Action: go west
                            Observation: Reservoir South. Wait two turns. Go north. Get trunk of jewels. Go south, southwest, southwest, west, south, up. Living Room. Put trunk in case.

                            Thought: Last treasure in this area is the Crystal Trident. Return to the Reservoir area, get the Air Pump, and proceed to the Atlantis Room for the Trident. Path: Down, North, East, North, Northeast, North.
                            Action: go down
                            Observation: Go north, east, north, northeast, and north. Go north again to Reservoir North. Get air pump. Go north again to Atlantis Room. Get crystal trident.

                            Thought: Now, return to the Living Room to deposit the Trident: Up, North, North, West, Down (Slide Room to Cellar), Up (to Living Room).
                            Action: go up
                            Observation: Go north, north, west, then down from Slide Room to Cellar. Go up. Living Room. Put trident in case.

                        (4) Temple Area and Hades Treasures
                            Thought: Need rope from the Attic to access the Torch Room. Path: East, Up, grab rope, then Down, West, Down to Cellar. Then North, East twice, Southeast, East to Dome Room.
                            Action: go east
                            Observation: Go up to the Attic. Get rope. Go down, west, and down to the Cellar. Go north, east 2 times, southeast, and east. Dome Room.

                            Thought: Tie the rope and descend. In the Torch Room, get the torch and turn off the lamp to conserve it. Proceed South to the Temple.
                            Action: tie rope to railing
                            Observation: Go down to the Torch Room. Get torch. Turn off lamp. Go south to the Temple. Drop all non-essential items (except torch).

                            Thought: Save due to Grue risk. Now go East to the Egyptian Room, get the coffin and sceptre. Then return West, South to Altar and 'pray' to teleport back to the surface to deposit the treasures.
                            Action: save game
                            Observation: Go east. Egyptian Room. Get coffin. Open coffin. Take sceptre. Go west, then south to Altar. Pray.

                            Thought: Teleported to Forest. Path back to house: East, South, East, In to Kitchen, West to Living Room. Deposit the coffin and sceptre. Then return to the Temple to continue the ritual.
                            Action: go east
                            Observation: Go south, east, in (Kitchen), west (Living Room). Put coffin and sceptre in case. Go down, north, east, east, southeast, east, down, south to Temple. Get items.

                            Thought: Back at the Temple, get the bell, candles, matchbook, and black book. The ritual for the Skull: Put out candles, check inventory (only ritual items), go Down twice to Entrance to Hades. Ring Bell.
                            Action: get matchbook
                            Observation: Get bell, candles, black book. Put out candles. Go down 2 times. Entrance to Hades. Ring bell.

                            Thought: Now, re-light the candles, read the book, and extinguish the candles. Then, proceed South to the Land of the Dead to get the skull.
                            Action: get candles
                            Observation: Light a match. Light candles. Read book. Put out candles. Drop book. Go south. Land of the Dead. Get crystal skull.

                            Thought: Skull acquired. Return to the Living Room to deposit the skull, and re-acquire the sceptre for the next section. Path: North, Up, North 3 times, West 2 times, South, Up.
                            Action: go north
                            Observation: Go up, north 3 times, west 2 times, south, and up. Living Room. Put skull in case. Get sceptre. Return to Temple to collect belongings.
                        
                        (5) Old Man River and Pot of Gold
                            Thought: Prepare for the boat journey. Path: South, Down, North 3 times, East, Up, East twice to Dam Base. Inflate the raft with the Air Pump and drop the pump.
                            Action: go south
                            Observation: Go down, north 3 times, east, up, east 2 times. Dam Base. Inflate pile of plastic with air pump. Drop pump. Read label. Drop label.

                            Thought: Put sharp items (screwdriver, sceptre) into the sack to avoid puncturing the raft. Get in the boat and launch. Wait four turns to drift.
                            Action: put screwdriver in sack
                            Observation: Put sceptre in sack. Get in boat. Launch. Wait four turns. Red buoy appears.

                            Thought: Grab the red buoy. The boat lands at Sandy Beach. Open the buoy for the emerald, then drop it. Grab the shovel.
                            Action: get red buoy
                            Observation: Sandy Beach. Stand up. Open buoy. Take emerald. Drop buoy. Get shovel.

                            Thought: Dig in the Sandy Cave four times for the scarab. Then return to the beach.
                            Action: go northeast
                            Observation: Sandy Cave. Dig in sand with shovel four times. Drop shovel and take the scarab. Go southwest. Sandy Beach.

                            Thought: If I have the sceptre, go South twice to Aragain Falls and wave it to cross the rainbow (West twice). If not, take the boat back to the west bank. Assuming I have it.
                            Action: go south
                            Observation: Go south again. Aragain Falls. Wave sceptre. Go west twice across the rainbow.

                            Thought: Too much weight for the pot of gold now. Deposit the scarab, emerald, and sceptre in the case first. Path: Southwest, Up, Up, Northwest, West twice to Living Room.
                            Action: go southwest
                            Observation: Go up, up, northwest, west 2 times. Living Room. Put scarab, emerald, and sceptre in case.

                            Thought: Now go back for the Pot of Gold. Path: East 4 times to Canyon View, then Down twice to End of Rainbow. Pick up the pot of gold (if the Thief hasn't stolen it).
                            Action: go east
                            Observation: Go east 3 times. Canyon View. Go down 2 times. End of Rainbow. Get pot of gold.

                            Thought: Return to the Living Room and deposit the final treasure from this section. Path: Southwest, Up, Up, Northwest, West 3 times.
                            Action: go southwest
                            Observation: Go up, up, northwest, west 3 times. Living Room. Put pot of gold in case.
                    
                        (6) Coal Mine, Final Treasures, and Ending
                            Thought: Time for the Coal Mine. Take the garlic out of the sack for the bat. Path: Down, North, East twice, South twice to the South Mirror Room. Touch the mirror to teleport.
                            Action: take garlic out of sack
                            Observation: Go down, north, east 2 times, south 2 times. South Mirror Room. Touch mirror.

                            Thought: Now in the North Mirror Room. Path: North, West, North, West, North to the Bat Room. The garlic should keep the bat at bay so I can get the jade figurine.
                            Action: go north
                            Observation: Go west, north, west, north. Bat Room. Get jade figurine (bat is holding his nose). Go east to Shaft Room.

                            Thought: Place non-flammable items and light sources in the basket to prepare for the Gas Room. I'll use the candles for the coal puzzle. Switch to the lantern.
                            Action: put candles in basket
                            Observation: Put screwdriver in basket. Drop torch. Turn on lantern.

                            Thought: Now, collect the bracelet and coal. Path: North, Down to Gas Room (bracelet), then East, Northeast, Southeast, Southwest, Down twice to Ladder Bottom (coal).
                            Action: go north
                            Observation: Go down. Gas Room. Get bracelet. Go east, northeast, southeast, southwest, down 2 times. Ladder Bottom. Go south. Get coal.

                            Thought: Return to the Shaft Room, load the coal, and the lit candles into the basket to ignite it. Path: North, Up twice, North, East, South, North, Up, South to Shaft Room.
                            Action: go north
                            Observation: Go up 2 times, north, east, south, north, up, south. Shaft Room. Put coal in basket. Get candles, light match, light candles, put candles in basket.

                            Thought: Lower the basket and return to the Timber Room to retrieve the processed coal (now diamond) and tools. Path: North, Down, East, Northeast, Southeast, Southwest, Down twice to Ladder Bottom. Go west twice to Drafty Room.
                            Action: lower basket
                            Observation: Go north, down, east, northeast, southeast, southwest, down 2 times. Ladder Bottom. Go west. Timber Room. Drop items. Go west. Drafty Room.

                            Thought: Get the coal (now diamond), screwdriver, and candles. Use the screwdriver to turn the switch on the machine in the Machine Room, which turns the coal into a diamond. Drop the screwdriver.
                            Action: get coal
                            Observation: Get screwdriver and candles from basket. Go south. Machine Room. Open machine, put coal inside, close it. Turn switch with screwdriver. Open machine. Get diamond. Drop screwdriver.

                            Thought: Return the diamond and candles to the Shaft Room, then raise the basket and retrieve them. Path: North, East, pick up items, and return to Shaft Room.
                            Action: go north
                            Observation: Put candles and diamond in basket. Go east. Pick up items (except timber). Return to Shaft Room. Raise basket. Take diamond and candles. Put out candles. Get torch.

                            Thought: Return to the Living Room to deposit all new treasures: Down, North, East, South, Down to Cellar, Up to Living Room. Deposit diamond, bracelet, figurine, and torch (if retrieved). Drop garlic/candles.
                            Action: go west
                            Observation: Go south, east, south, down. Cellar. Go up. Living Room. Put diamond, bracelet, figurine, and torch in case. Drop garlic and candles.

                            Thought: Final confrontation. Get the nasty knife from the Attic. Go to the Cyclops Room (via West twice or through the Maze), and then Up to the Treasure Room to find and attack the Thief.
                            Action: go east
                            Observation: Go up. Attic. Pick up nasty-looking knife. Return down and west. Go west 2 times (or Maze route) to Cyclops Room. Go up. Treasure Room. Attack Thief with nasty knife.

                            Thought: Thief defeated. Collect the silver chalice, the opened egg (now with the canary), and any stolen treasures. Return to the Living Room and deposit them, winding the canary outside for the brass bauble.
                            Action: take silver chalice
                            Observation: Take opened egg/canary. Take stolen treasures. Return to Living Room. Turn off lamp. Put chalice in case. Get canary from egg. Put egg in case. Put stolen treasures in case. Go east 2 times, north 2 times. Forest Path. Wind canary. Get brass bauble. Return to Living Room. Put canary and bauble in case.

                            Thought: Check if all treasures are in the case. The final goal is the Stone Barrow. Check the trophy case for the parchment, read it, and follow the secret path it reveals.
                            Action: look at trophy case
                            Observation: An "ancient parchment" appears. Take it and look at it. Go east 2 times, south, and west to West of House. Go southwest (down the "secret path") then enter the Stone Barrow to the west.
                        
                    """

                agent_response = await self.agent.ainvoke(
                    {
                        "llm": model,
                        "template": template,
                        "history": [],
                        "history_max_length": 200,
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
        category = "react_prompting"
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
