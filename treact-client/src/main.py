import asyncio

from treact_client import prompt_template, run_client


def standard_client() -> None:
    asyncio.run(
        run_client(
            "Standard",
            prompt_template=prompt_template.STANDARD,
            game_name="zork1",
            maximum_step=250,
            missing_tool_call_threshold=5,
            history_max_length=10,
            max_branch_per_node=1,
            iterations=10,
            config={"recursion_limit": 1200},
        )
    )


def react_client() -> None:
    asyncio.run(
        run_client(
            "React",
            prompt_template=prompt_template.REACT,
            game_name="zork1",
            maximum_step=250,
            missing_tool_call_threshold=5,
            history_max_length=10,
            max_branch_per_node=1,
            iterations=10,
            config={"recursion_limit": 1200},
        )
    )


def action_client() -> None:
    asyncio.run(
        run_client(
            "Action",
            prompt_template=prompt_template.ACTION,
            game_name="zork1",
            maximum_step=250,
            missing_tool_call_threshold=5,
            history_max_length=10,
            max_branch_per_node=1,
            iterations=10,
            config={"recursion_limit": 1200},
        )
    )


def treact_client() -> None:
    asyncio.run(
        run_client(
            "Action",
            prompt_template=prompt_template.TREACT,
            game_name="zork1",
            maximum_step=250,
            missing_tool_call_threshold=5,
            history_max_length=10,
            max_branch_per_node=3,
            iterations=10,
            config={"recursion_limit": 1200},
        )
    )
