from fastapi import FastAPI
from zork_api import ZorkInstance
from pydantic import BaseModel, Field
import uvicorn


class CommandRequest(BaseModel):
    commands: list[str] = Field(
        default=[],
        description="List of commands to execute in Zork",
        examples=["look", "go north", "take lamp"],
    )


def zork_post(commands: list[str]) -> str:
    if not commands:
        with ZorkInstance("zork_285.z5") as zork:
            return zork.initial_response
    with ZorkInstance("zork_285.z5") as zork:
        response = ""
        for command in commands:
            response = zork.send_command(command)
    return response


app = FastAPI()


@app.post("/zork/zork285")
def zork285(request: CommandRequest):
    print(request.commands)
    return {"response": zork_post(request.commands)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
