"""Offline tests for the sandbox runner and post-snippet board refresh."""

import base64
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assignment.agent.chess_agent import ChessAgent
from assignment.agent.chess_tools import _run_python
from assignment.agent.tools import PLAY_MOVE_TOOL, RUN_PYTHON_TOOL, SIMULATE_MOVE_TOOL


def test_run_python_returns_runner_stdout_unchanged():
    output = json.dumps({"stdout": "hello\n", "stderr": "", "error": None})
    env = SimpleNamespace(execute=Mock(return_value={
        "stdout": output, "stderr": "warning", "returncode": 0,
        "exception_info": "",
    }))
    code = 'print("hello")'
    assert _run_python(env, 8000, json.dumps({"code": code})) == output
    env.execute.assert_called_once_with(
        ["python", "/opt/assignment/sandbox_python.py", "8000",
         base64.b64encode(code.encode()).decode()],
        shell=False,
    )


def test_snippet_exception_is_not_a_tool_error():
    output = json.dumps({
        "stdout": "", "stderr": "Traceback...", "error": "ValueError: bad",
    })
    env = SimpleNamespace(execute=Mock(return_value={
        "stdout": output, "stderr": "", "returncode": 0, "exception_info": "",
    }))
    assert _run_python(env, 8000, '{"code": "raise ValueError()"}') == output


@pytest.mark.parametrize("arguments", ["{bad", "[]", "{}", '{"code": 123}'])
def test_invalid_runner_arguments_do_not_execute(arguments):
    env = SimpleNamespace(execute=Mock())
    assert _run_python(env, 8000, arguments).startswith("<chess_error>")
    env.execute.assert_not_called()


@pytest.mark.parametrize("result, expected", [
    ({"returncode": 1, "stderr": "runner failed"}, "runner failed"),
    ({"returncode": -1, "exception_info": "timeout"}, "timeout"),
])
def test_sandbox_failure_returns_error(result, expected):
    env = SimpleNamespace(execute=Mock(return_value=result))
    assert _run_python(env, 8000, '{"code": "pass"}') == (
        f"<chess_error>{expected}</chess_error>"
    )


@pytest.mark.parametrize("runner_result", [
    '{"stdout": "output", "stderr": "", "error": null}',
    '<chess_error>Sandbox execution failed</chess_error>',
])
@pytest.mark.parametrize("refresh_fails", [False, True])
def test_dispatch_always_refreshes_and_preserves_output(
    monkeypatch, runner_result, refresh_fails,
):
    # Isolate dispatch from environment startup and model API initialization.
    agent = ChessAgent.__new__(ChessAgent)
    agent.env = object()
    agent.chess_client = object()
    agent.python_sandbox_port = 8000
    agent.programmatic_tools = True
    agent.tools = [PLAY_MOVE_TOOL, RUN_PYTHON_TOOL, SIMULATE_MOVE_TOOL]
    agent.skills = {}
    agent.last_state = {"game_over": False}
    agent.finished = False
    agent.format_state = Mock(return_value="<chess_state>fresh</chess_state>")
    state = {"game_over": True}
    run = Mock(return_value=runner_result)
    refresh = Mock(
        side_effect=RuntimeError("offline") if refresh_fails else None,
        return_value=state,
    )
    monkeypatch.setattr("assignment.agent.chess_agent._run_python", run)
    monkeypatch.setattr("assignment.agent.chess_agent._game_state", refresh)
    arguments = '{"code": "pass"}'
    messages = agent.execute_tool_calls([{
        "id": "python_1", "type": "function",
        "function": {"name": "run_python", "arguments": arguments},
    }])
    run.assert_called_once_with(agent.env, 8000, arguments)
    refresh.assert_called_once_with(agent.chess_client, reset=False)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "python_1"
    assert messages[0]["content"].startswith(runner_result + "\n\n")
    if refresh_fails:
        assert "<chess_error>Failed to retrieve game state: offline</chess_error>" in messages[0]["content"]
        assert agent.finished is False
    else:
        assert messages[0]["content"].endswith("<chess_state>fresh</chess_state>")
        assert agent.last_state == state
        assert agent.finished is True
