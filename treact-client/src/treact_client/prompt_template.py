SUMMARIZE = """
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
