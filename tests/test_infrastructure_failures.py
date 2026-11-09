"""Infrastructure failure must abort, not consume the model step budget."""

import base64
import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from assignment.agent.base import Agent
from assignment.agent.chess_agent import ChessAgent
from assignment.agent.chess_tools import _play_move, _run_python, _simulate_move
from assignment.agent.tools import RUN_PYTHON_TOOL
from assignment.chess_sandbox import SANDBOX_FILES
from assignment.env import Environment
from assignment.errors import InfrastructureError


@pytest.mark.parametrize("tool,arguments", [
    (_play_move, '{"move": "e2e4"}'),
    (_simulate_move, '{"fen": "test"}'),
])
def test_transport_failure_is_bounded(tool, arguments):
    handler = Mock(side_effect=httpx.ConnectError("SSL failure"))
    with httpx.Client(base_url="http://chess.test", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(InfrastructureError, match="transport failure"):
            tool(client, arguments)
    assert handler.call_count == 3
    if tool is _play_move:
        keys = [call.args[0].headers["Idempotency-Key"] for call in handler.call_args_list]
        assert len(set(keys)) == 1


@pytest.mark.parametrize("response", [
    httpx.Response(503, text="Service unavailable"),
    httpx.Response(500, json={"detail": "Internal error"}),
])
def test_server_failure_is_not_reported_as_an_illegal_move(response):
    with httpx.Client(
        base_url="http://chess.test", transport=httpx.MockTransport(lambda _: response),
    ) as client:
        with pytest.raises(InfrastructureError, match="HTTP 50"):
            _play_move(client, '{"move": "e2e4"}')


def test_dead_sandbox_is_not_converted_to_tool_error():
    env = SimpleNamespace(execute=Mock(side_effect=InfrastructureError("sandbox is dead")))
    with pytest.raises(InfrastructureError, match="sandbox is dead"):
        _run_python(env, 8000, '{"code": "pass"}')


def test_refresh_infrastructure_failure_propagates(monkeypatch):
    agent = ChessAgent.__new__(ChessAgent)
    agent.tools = [RUN_PYTHON_TOOL]
    agent.env = Mock()
    agent.python_sandbox_port = 8000
    agent.chess_client = Mock()
    monkeypatch.setattr("assignment.agent.chess_agent._run_python", Mock(return_value="output"))
    monkeypatch.setattr("assignment.agent.chess_agent._game_state", Mock(side_effect=InfrastructureError("offline")))
    with pytest.raises(InfrastructureError, match="offline"):
        agent.execute_tool_calls([{
            "id": "python", "function": {"name": "run_python", "arguments": '{"code": "pass"}'},
        }])


def test_run_aborts_first_step_and_preserves_error_and_trajectory(tmp_path):
    agent = Agent.__new__(Agent)
    agent.step_limit = 200
    agent.finished = False
    agent.maybe_compact_context = Mock()
    agent.query_language_model = Mock(return_value={"tool_calls": [{"id": "move"}]})
    agent.execute_tool_calls = Mock(side_effect=InfrastructureError("offline"))
    agent.assistant_messages = []
    agent.api_prompts = []
    agent.api_responses = []
    agent.compaction_events = []
    agent.logs_save_path = str(tmp_path / "trajectory.json")
    agent.auto_stop_environment = True
    agent.env = SimpleNamespace(stop=Mock(side_effect=RuntimeError("cleanup failed")))
    with pytest.raises(InfrastructureError, match="offline"):
        agent.run()
    agent.query_language_model.assert_called_once()
    agent.env.stop.assert_called_once()
    assert json.loads((tmp_path / "trajectory.json").read_text())["compactions"] == []


@pytest.mark.parametrize("task_error", [None, ValueError("task failed")])
def test_context_cleanup_preserves_task_exception(task_error):
    env = Environment.__new__(Environment)
    env.stop = Mock(side_effect=RuntimeError("cleanup failed"))
    if task_error is None:
        with pytest.raises(RuntimeError, match="cleanup failed"):
            env.__exit__(None, None, None)
    else:
        assert env.__exit__(type(task_error), task_error, None) is False


def test_shipped_runner_imports_without_assignment_package(tmp_path):
    # Reproduce the flat /opt/assignment layout, not the host package layout.
    for source in SANDBOX_FILES:
        shutil.copy(source, tmp_path / source.name)
    code = base64.b64encode(b'print("runner works")').decode()
    result = subprocess.run([
        sys.executable, "-I", "-c",
        f"import sys, runpy; sys.path.insert(0, {str(tmp_path)!r}); "
        f"sys.argv = ['sandbox_python.py', '8000', {code!r}]; "
        f"runpy.run_path({str(tmp_path / 'sandbox_python.py')!r}, run_name='__main__')",
    ], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == {
        "stdout": "runner works\n", "stderr": "", "error": None,
    }


def test_simulation_retries_transient_disconnect():
    handler = Mock(side_effect=[
        httpx.RemoteProtocolError("Server disconnected"),
        httpx.Response(200, json={"fen": "unchanged"}),
    ])
    with httpx.Client(base_url="http://chess.test", transport=httpx.MockTransport(handler)) as client:
        assert json.loads(_simulate_move(client, '{"fen": "test"}')) == {"fen": "unchanged"}
    assert handler.call_count == 2

