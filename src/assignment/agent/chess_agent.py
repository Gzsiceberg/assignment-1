"""The Part 3 chess agent: reuse the shared loop with one new domain tool."""

from __future__ import annotations

import json
from typing import Any

import httpx

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
)
from assignment.agent.chess_tools import (
    _game_state,
    _invoke_skill,
    _play_move,
    CHESS_PORT,
    _run_python,
    _simulate_move,
)
from assignment.agent.tools import (
    INVOKE_SKILL_TOOL,
    PLAY_MOVE_TOOL,
    RUN_PYTHON_TOOL,
    SIMULATE_MOVE_TOOL,
)
from assignment.prompts import (
    CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE,
    CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE,
    PROGRAMMATIC_CHESS_PROMPT,
)
from assignment.env import Environment


def format_chess_state(
    state: dict[str, Any], *, include_legal_moves: bool = True
) -> str:
    """Turn chess API JSON into a compact observation an LLM can act on."""

    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")

    recent_history = state.get("history", [])[-8:]
    history = (
        " ".join(
            f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history
        )
        or "(none)"
    )
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    board_text = "\n".join(board_lines)

    observation = (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n"
        f"{board_text}\n"
        f"recent_history: {history}\n"
    )
    if include_legal_moves:
        observation += f"legal_moves: {legal_moves}\n"
    return observation + "</chess_state>"


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 200,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        http_client: Any | None = None,
        reset_game: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None,
        programmatic_tools: bool = False,
        python_sandbox_port: int = CHESS_PORT,
        include_legal_moves: bool = True,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )

        # TODO(Part 3): Register the play_move tool schema from tools.py.
        self.tools.append(PLAY_MOVE_TOOL)
        self.tools.append(SIMULATE_MOVE_TOOL)

        if programmatic_tools:
            self.tools.append(RUN_PYTHON_TOOL)

        # run_python always executes in the sandbox, on the port the chess
        # server is listening on there.
        self.python_sandbox_port = python_sandbox_port
        self.include_legal_moves = include_legal_moves

        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError(
                    "ChessAgent needs an environment with server_url or an http_client."
                )
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client

        initial_state = _game_state(self.chess_client, reset=reset_game)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        prompt_template = (
            CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE
            if include_legal_moves
            else CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE
        )
        self.system_prompt = prompt_template.render()
        if programmatic_tools:
            self.system_prompt += "\n\n" + PROGRAMMATIC_CHESS_PROMPT
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )
        opening_instruction = (
            "Choose one move from legal_moves and call play_move."
            if include_legal_moves
            else "Infer a legal move from the board and FEN, then call play_move."
        )
        self.task_prompt = (
            f"Play this game as White. {opening_instruction}\n\n"
            f"{self.format_state(initial_state)}"
        )
        self.programmatic_tools = programmatic_tools

    def format_state(self, state: dict[str, Any]) -> str:
        """Format a state according to this run's observation condition."""

        return format_chess_state(state, include_legal_moves=self.include_legal_moves)

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute model-generated ``play_move`` calls against the chess API."""

        # TODO(Part 3.1):
        # 1. Dispatch on the function name, and ignore a tool this agent did
        #    not register.
        # 2. Hand the raw arguments to the matching chess_tools helper, with
        #    self.chess_client as its first argument. Each helper takes the
        #    client explicitly so the same code can run inside the sandbox.
        # 3. Format a played move with self.format_state, then update
        #    last_state and finished.
        # 4. Link every observation to its call with tool_call_id.
        # 5. Turn malformed, unknown, rejected, or extra parallel calls into
        #    recoverable <chess_error> observations instead of crashing.
        valid_tool_names = [PLAY_MOVE_TOOL["function"]["name"]]
        if self.skills:
            valid_tool_names.append(INVOKE_SKILL_TOOL["function"]["name"])
        valid_tool_names.append(SIMULATE_MOVE_TOOL["function"]["name"])
        if self.programmatic_tools:
            valid_tool_names.append(RUN_PYTHON_TOOL["function"]["name"])

        tool_results = []
        has_play_move = False
        for tool_call in tool_calls:
            tool_function = tool_call.get("function", {})
            tool_name = tool_function.get("name")
            tool_call_id = tool_call.get("id")
            arguments = tool_function.get("arguments", "{}")
            if not tool_call_id:
                tool_results.append(
                    {
                        "role": "tool",
                        "content": f"Tool call is missing an id: {tool_call}"
                    }
                )
                continue

            if tool_name not in valid_tool_names:
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"<chess_error>Unknown tool: {tool_name}</chess_error>",
                    }
                )
                continue

            if tool_name == PLAY_MOVE_TOOL["function"]["name"]:
                if has_play_move:
                    result = "<chess_error>Multiple play_move calls in one response are not allowed.</chess_error>"
                else:
                    result = _play_move(self.chess_client, arguments)
                    has_play_move = True
                if "<chess_error>" not in result:
                    try:
                        new_state = json.loads(result)
                        self.last_state = new_state
                        self.finished = bool(new_state.get("game_over"))
                        result = self.format_state(new_state)
                    except Exception as exc:
                        result = f"<chess_error>Failed to parse play_move result: {exc}</chess_error>"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result,
                    }
                )

            # TODO(Part 3.3-4): add cases for simulate_move and run_python, with
            # linked observations and recoverable errors, just like the old tool.
            if tool_name == SIMULATE_MOVE_TOOL["function"]["name"]:
                result = _simulate_move(self.chess_client, arguments)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result,
                    }
                )

            if tool_name == INVOKE_SKILL_TOOL["function"]["name"]:
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": _invoke_skill(self.skills, arguments),
                    }
                )

            if tool_name == RUN_PYTHON_TOOL["function"]["name"]:
                result = _run_python(self.env, self.python_sandbox_port, arguments)
                # A snippet may commit a move before failing, so always refresh.
                try:
                    new_state = _game_state(self.chess_client, reset=False)
                    observation = self.format_state(new_state)
                    self.last_state = new_state
                    self.finished = bool(new_state.get("game_over"))
                    result += "\n\n" + observation
                except Exception as exc:
                    result += f"\n\n<chess_error>Failed to retrieve game state: {exc}</chess_error>"
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result,
                    }
                )
        return tool_results
