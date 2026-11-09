"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

if __package__:
    from assignment.errors import InfrastructureError
else:
    # This file is also shipped as a standalone module beside the runner.
    from errors import InfrastructureError

CHESS_PORT = 8000


class ToolArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class PlayMoveArguments(ToolArguments):
    move: str = Field(
        ...,
        description="The move to play in UCI notation, for example e2e4.",
    )


class SimulateMoveArguments(ToolArguments):
    fen: str = Field(
        ...,
        description="The FEN of the position to simulate the move from.",
    )
    move: str | None = Field(
        default=None,
        description="The move to simulate in UCI notation, for example e2e4. If not provided, only the FEN will be returned.",
    )


class InvokeSkillArguments(ToolArguments):
    name: str


class RunPythonArguments(ToolArguments):
    code: str = Field(
        ...,
        description="The Python code to run in the sandbox.",
    )


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    # The extension deduplicates moves carrying a request ID. Reuse that ID
    # across retries; never replay a reset or an unprotected mutation.
    safe = method == "GET" or endpoint == "/api/simulate" or (
        endpoint == "/api/move" and kwargs.get("headers", {}).get("Idempotency-Key")
    )
    attempts = 3 if safe else 1
    for attempt in range(attempts):
        try:
            response = client.request(method, endpoint, **kwargs)
            break
        except httpx.RequestError as exc:
            if attempt + 1 == attempts:
                raise InfrastructureError(
                    f"Chess transport failure after {attempts} attempt(s): {exc}"
                ) from exc
            time.sleep(0.25 * (attempt + 1))
    if response.status_code >= 500:
        raise InfrastructureError(f"Chess server returned HTTP {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run. Transport failures
    are fatal infrastructure errors, not mistakes the model can fix.
    """
    # TODO(Part 3.3.b): Parse the arguments, call the provided
    # /api/simulate endpoint with fen and optional move, and return its JSON.
    # Catch any errors raised by the tool and return an error message between
    # `<chess_error></chess_error>` for the agent to address. Cover malformed
    # JSON arguments, arguments that are not an object, a missing or
    # non-string fen, a non-string move, a position or move the server rejects,
    # and a transport failure.
    try:
        args = SimulateMoveArguments.model_validate_json(arguments)
    except ValidationError as exc:
        return f"<chess_error>Invalid arguments: {exc}</chess_error>"

    try:
        payload = _request_state(client, "POST", "/api/simulate",
                                 json=args.model_dump(exclude_none=True))
        return json.dumps(payload)
    except ValueError as exc:
        return f"<chess_error>Illegal move: {exc}</chess_error>"
    except RuntimeError as exc:
        return f"<chess_error>Server error: {exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move is rejected. Transport failures
    use a stable request ID for bounded, deduplicated retries.
    """
    # TODO(3.1.b): Parse the arguments and POST {"move": <uci move>} to
    # /api/move. Return its JSON object. Catch any errors raised by the
    # tool and return an error message between `<chess_error></chess_error>`
    # for the agent to address. Cover malformed JSON arguments, arguments
    # that are not an object, a missing or non-string fen, a non-string move,
    # a position or move the server rejects, and a transport failure.
    try:
        args = PlayMoveArguments.model_validate_json(arguments)
    except ValidationError as exc:
        return f"<chess_error>Invalid arguments: {exc}</chess_error>"

    try:
        payload = _request_state(
            client, "POST", "/api/move", json=args.model_dump(),
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        return json.dumps(payload)
    except ValueError as exc:
        return f"<chess_error>Illegal move: {exc}</chess_error>"
    except RuntimeError as exc:
        return f"<chess_error>Server error: {exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    # TODO(3.4): parse the arguments and run the code in the
    # sandbox with the registered tools available by name.
    #
    # `/opt/assignment/sandbox_python.py` is a script on the `env` sandbox
    # that has access to the same tool definitions in this file. Use it to run
    # the code that the model produced as an argument to the run_python tool.
    # The script accepts two positional arguments -- `port` and a base64-encoded
    # string of code (to prevent issues with quoting). Implement this tool
    # call.
    #
    # The script prints one JSON object with `stdout`, `stderr`, and `error`
    # from running the code -- return that string as it is.
    #
    # A non-zero returncode means the sandbox itself failed, not the model's
    # code. Report `exception_info` or `stderr` as a <chess_error>.
    #
    # Return <chess_error>{message}</chess_error> if there are issues like type
    # mismatches or parsing failures.
    try:
        args = RunPythonArguments.model_validate_json(arguments)
    except ValidationError as exc:
        return f"<chess_error>Invalid arguments: {exc}</chess_error>"

    try:
        code_b64 = base64.b64encode(args.code.encode()).decode()
        result = env.execute(
            ["python", "/opt/assignment/sandbox_python.py", str(port), code_b64],
            shell=False,
        )
        if result.get("returncode") != 0 or result.get("exception_info"):
            message = (
                result.get("exception_info")
                or result.get("stderr")
                or f"Sandbox command failed with return code {result.get('returncode')}"
            )
            return f"<chess_error>{message}</chess_error>"
        return result["stdout"]
    except InfrastructureError:
        raise
    except Exception as exc:
        return f"<chess_error>Sandbox execution error: {exc}</chess_error>"


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    try:
        args = InvokeSkillArguments.model_validate_json(arguments)
        if args.name not in skills:
            return f"<chess_error>Unknown skill: {args.name}</chess_error>"
        content = skills[args.name]["content"]
        if not isinstance(content, str):
            raise TypeError("Skill content must be a string.")
        return content
    except (ValidationError, KeyError, TypeError) as exc:
        return f"<chess_error>Invalid invoke_skill request: {exc}</chess_error>"


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
