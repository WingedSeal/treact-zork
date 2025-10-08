import argparse
from pathlib import Path

from treact_client import run_client_file

print("Running 'main.py'")


def standard_client() -> None:
    run_client_file(Path("./clients/standard.toml"))


def react_client() -> None:
    run_client_file(Path("./clients/react.toml"))


def action_client() -> None:
    run_client_file(Path("./clients/action.toml"))


def treact_client() -> None:
    run_client_file(Path("./clients/treact.toml"))


def custom_client() -> None:
    parser = argparse.ArgumentParser(
        description="Run MCP-Client with a configuration file"
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        default="client.toml",
        help="Path to the configuration file (default: client.toml)",
    )
    args = parser.parse_args()
    config_path = Path(args.config_file)
    run_client_file(config_path)
