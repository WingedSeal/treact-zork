from pathlib import Path
from typing import Any
import tomllib
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel


class ClientConfig(BaseModel):
    name: str
    prompt_template: str
    game_name: str
    maximum_step: int
    missing_tool_call_threshold: int
    history_max_length: int
    max_branch_per_node: int


class ExecutionConfig(BaseModel):
    iterations: int


class TOMLConfig(BaseModel):
    client: ClientConfig
    config: RunnableConfig
    execution: ExecutionConfig


def parse_toml_config(toml_path_str: str) -> dict[str, Any]:
    toml_path = Path(toml_path_str)

    if not toml_path.exists():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    with open(toml_path, "rb") as f:
        raw_data = tomllib.load(f)

    config = TOMLConfig(**raw_data)

    template_path = toml_path.parent / config.client.prompt_template
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    prompt_template_content = template_path.read_text(encoding="utf-8")

    return {
        "client_name": config.client.name,
        "prompt_template": prompt_template_content,
        "game_name": config.client.game_name,
        "maximum_step": config.client.maximum_step,
        "missing_tool_call_threshold": config.client.missing_tool_call_threshold,
        "history_max_length": config.client.history_max_length,
        "max_branch_per_node": config.client.max_branch_per_node,
        "iterations": config.execution.iterations,
        "config": config.config,
    }
