from pydantic import BaseModel, Field


class AIModelResponse(BaseModel):
    """
    Represents the structured response from the AI model regarding the Zork game state.
    """

    game_completed: bool = Field(
        description="True if the Zork game has been successfully completed by collecting all 20 treasures "
        "and achieving victory. False if the game is still in progress or if the player has died. "
        "Only set to True when the game explicitly indicates victory/winning condition met."
    )
    current_status: str = Field(
        description="Current game state description including: current location, recent game response, "
        "inventory status, immediate objectives, or any important game feedback. "
        "This helps track progress and inform the next decision."
    )

    score: int = Field(
        description="Current score in the Zork game, representing the number of treasures collected so far."
    )
