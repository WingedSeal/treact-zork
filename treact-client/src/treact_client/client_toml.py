import tomllib
from pathlib import Path
from typing import TypedDict

from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel


class ClientConfig(BaseModel):
    name: str
    prompt_template: str
    game_name: str
    maximum_step: int
    len_tool_calls_out_of_range_threshold: int
    history_max_length: int
    max_branch_per_node: int
    min_tool_calls: int
    max_tool_calls: int


class ExecuteConfig(BaseModel):
    iterations: int


class TOMLConfig(BaseModel):
    client: ClientConfig
    config: RunnableConfig
    execute: ExecuteConfig


class TOMLTypedDict(TypedDict):
    client_name: str
    prompt_template: str
    game_name: str
    maximum_step: int
    len_tool_calls_out_of_range_threshold: int
    history_max_length: int
    max_branch_per_node: int
    min_tool_calls: int
    max_tool_calls: int
    iterations: int
    config: RunnableConfig


def parse_toml_config(toml_path: Path) -> TOMLTypedDict:

    if not toml_path.exists():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    with toml_path.open("rb") as f:
        raw_data = tomllib.load(f)

    config = TOMLConfig(**raw_data)

    raw_path = Path(config.client.prompt_template)
    if raw_path.is_absolute():
        template_path = raw_path
    else:
        template_path = toml_path.parent / raw_path

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    prompt_template_content = template_path.read_text(encoding="utf-8")

    return {
        "client_name": config.client.name,
        "prompt_template": prompt_template_content,
        "game_name": config.client.game_name,
        "maximum_step": config.client.maximum_step,
        "len_tool_calls_out_of_range_threshold": config.client.len_tool_calls_out_of_range_threshold,
        "history_max_length": config.client.history_max_length,
        "max_branch_per_node": config.client.max_branch_per_node,
        "min_tool_calls": config.client.min_tool_calls,
        "max_tool_calls": config.client.max_tool_calls,
        "iterations": config.execute.iterations,
        "config": config.config,
    }
