from key_manager import KEY_LENGTH, KeyManager, key_example
import uvicorn
from fastapi import FastAPI
from zork_api import ZorkInstance
from pydantic import BaseModel, Field


class CommandRequest(BaseModel):
    command: str = Field(
        default=[], description="Command to execute in Zork", examples=["look"]
    )
    key: str = Field(
        description=f"{KEY_LENGTH} characters string for accessing Zork session",
        examples=[key_example],
    )


class CommandsRequest(BaseModel):
    commands: list[str] = Field(
        default=[],
        description="List of commands to execute in Zork",
        examples=[["look", "go north", "take lamp"]],
    )


def zork_post(commands: list[str], zork_file: str) -> str:
    if not commands:
        with ZorkInstance("games/" + zork_file) as zork:
            return zork.initial_response
    with ZorkInstance("games/" + zork_file) as zork:
        response = ""
        for command in commands:
            response = zork.send_command(command)
    return response


app = FastAPI()
app.state.key_manager = KeyManager(["zork285"])


@app.post("/zork/zork285")
def zork285(request: CommandsRequest):
    return {"response": zork_post(request.commands, "zork_285.z5")}


@app.get("/gen_key/zork285")
def gen_key_zork285():
    return {
        "initial_response": zork_post([], "zork_285.z5"),
        "key": app.state.key_manager.gen_key("zork285"),
    }


@app.post("/use_key/zork285")
def use_key_zork285(request: CommandRequest):
    new_key = app.state.key_manager.add_command("zork285", request.key, request.command)
    if not new_key:
        return {"key_valid": False, "response": "", "new_key": ""}
    return {
        "key_valid": True,
        "response": zork_post(
            app.state.key_manager.get_history("zork285", new_key), "zork_285.z5"
        ),
        "new_key": new_key,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
