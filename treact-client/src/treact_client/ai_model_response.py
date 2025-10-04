from pydantic import BaseModel, Field


class AIModelResponse(BaseModel):
    """
    Represents the structured response from the AI model regarding the Zork game state.
    """

    game_completed: bool = Field(
        description="""True if the Zork game has been successfully completed by collecting all treasures and putting them in the trophy case in the Living Room of the house.
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
        False if the game is still in progress or if the player has died.
        Only set to True when the game explicitly indicates victory/winning condition met."""
    )
    current_inventory: str = Field(
        description="A comma-separated list of items currently held in the player's inventory."
    )
    current_status: str = Field(
        description="Current game state description including: current location, recent game response, "
        "inventory status, immediate objectives, or any important game feedback. "
        "This helps track progress and inform the next decision."
    )

    score: int = Field(
        description="Current score in the Zork game, representing the number of treasures collected so far."
    )

    move: int = Field(
        description="The current move number in the Zork game, indicating how many valid actions have been taken."
    )
