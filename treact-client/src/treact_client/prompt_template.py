SUMMARIZE = """
Summarize the following history of commands and responses from playing Zork in the following JSON format.
{{
    game_completed: bool,
    current_status: str,
    score: int
}}
- game_completed: Whether the game explicitly indicates victory/winning and player has completed the game by collecting all 20 treasures.
- current_status: Current location, Inventory, and important game feedbacks.
- score: Current score in the game representing number of treasures collected.

** Example **
{{
    "game_completed": false,
    "current_status": "You are in the Living Room. You have 2 treasures: a lamp and a sword. Your score is 20 out of 350. You need to find more treasures and avoid the troll.",
    "score": 20
}}

** History **
{tool_call_result_history}
"""

_SUMMARIZE = """
Summarize the following history of commands and responses from playing zork in Json format with the following keys:
1. game_completed: True if the Zork game has been successfully completed by collecting all 20 treasures and achieving victory. False if the game is still in progress or if the player has died. Only set to True when the game explicitly indicates victory/winning condition met.
2. current_status: Current game state description including: current location, recent game response, inventory status, immediate objectives, or any important game feedback. This helps track progress and inform the next decision.
3. score: Current score in the Zork game, representing the number of treasures collected so far.

* Please return your response as valid JSON with this exact structure *
{{
    "game_completed": false,
    "current_status": "You are in the Living Room. You have 2 treasures: a lamp and a sword. Your score is 20 out of 350. You need to find more treasures and avoid the troll.",
    "score": 20
}}

** History **
    {tool_call_result_history}
"""

PRUNE_HISTORY = """
Select the most promising tool call results for continuing the Zork game.

** Previous thought: {thought}

** Tool call results (with indices) **
{tool_call_result_with_indices}

** Instructions **
Select UP TO {max_branch_per_node} indices that show the best progress toward collecting treasures.
Focus on results that:
- Advance toward treasures
- Discover new locations or items
- Provide useful game information

Return as a list of integers: [index1, index2, ...]
"""

FINAL_NODE = """
From the following leaf nodes representing different possible game states in Zork, select the node that maximizes the game's progress based on the current score and game completion status.

** Leaf Nodes (with indices) **
{leaf_nodes_with_indices}

Please select the index (0-based) of the leaf node that represents the best game state.
Consider factors like:
- Current score (higher is better)
- Progress toward treasures
- Strategic position in the game
- Information gained

Return the index as an integer (e.g., 0 for the first node, 1 for the second, etc.)
"""