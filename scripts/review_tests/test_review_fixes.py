"""Offline regressions kept outside the assignment's protected tests/ tree."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from openai.types.chat import ChatCompletion
import pytest

from assignment.agent.base import Agent, StepLimitError
from assignment.agent.chess_agent import ChessAgent
from assignment.agent.tools import PLAY_MOVE_TOOL, RUN_PYTHON_TOOL, SIMULATE_MOVE_TOOL


def completion(content=None, tool_calls=None, finish_reason="stop"):
    return ChatCompletion.model_validate({
        "id": "offline", "object": "chat.completion", "created": 0,
        "model": "offline",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": content, "tool_calls": tool_calls,
        }}],
    })


def call(name, arguments="{}", call_id="repeated_id"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": arguments,
    }}


class FinishingAgent(Agent):
    def execute_tool_calls(self, tool_calls):
        self.finished = True
        return [{"role": "tool", "tool_call_id": c["id"], "content": "done"}
                for c in tool_calls]


def loop_agent(tmp_path, responses, step_limit):
    agent = FinishingAgent.__new__(FinishingAgent)
    agent.system_prompt = "system"
    agent.task_prompt = "task"
    agent.assistant_messages = []
    agent.api_prompts = []
    agent.api_responses = []
    agent.compaction_events = []
    agent.compact_threshold_tokens = None
    agent.steps_taken = 0
    agent.step_limit = step_limit
    agent.finished = False
    agent.tools = []
    agent.model = "offline"
    agent.logs_save_path = str(tmp_path / "trajectory.json")
    agent.auto_stop_environment = True
    agent.env = SimpleNamespace(stop=Mock())
    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=Mock(side_effect=responses),
    )))
    return agent


def test_stop_text_is_preserved_and_recovered_before_domain_completion(tmp_path):
    agent = loop_agent(tmp_path, [
        completion("I will use a tool."),
        completion(tool_calls=[call("finish")], finish_reason="tool_calls"),
    ], 2)
    agent.run()
    assert agent.finished is True
    assert agent.steps_taken == 2
    second_prompt = agent.api_prompts[1]
    assert second_prompt[2] == {"role": "assistant", "content": "I will use a tool."}
    assert second_prompt[3]["role"] == "user"
    agent.env.stop.assert_called_once()
    saved = json.loads(open(agent.logs_save_path).read())
    assert len(saved["responses"]) == 2


def test_repeated_text_still_obeys_step_limit_and_cleanup(tmp_path):
    agent = loop_agent(tmp_path, [completion("No action")] * 2, 2)
    with pytest.raises(StepLimitError):
        agent.run()
    assert agent.steps_taken == 2
    assert agent.finished is False
    agent.env.stop.assert_called_once()
    assert len(json.loads(open(agent.logs_save_path).read())["responses"]) == 2


def test_already_finished_agent_does_not_request_an_action(tmp_path):
    agent = loop_agent(tmp_path, [], 2)
    agent.finished = True
    agent.run()
    agent.client.chat.completions.create.assert_not_called()
    agent.env.stop.assert_called_once()


@pytest.mark.parametrize("names", [
    ("play_move", "play_move"),
    ("play_move", "run_python"),
    ("run_python", "play_move"),
    ("run_python", "run_python"),
])
@pytest.mark.parametrize("first_fails", [False, True])
def test_parallel_live_actions_share_one_guard(monkeypatch, names, first_fails):
    agent = ChessAgent.__new__(ChessAgent)
    agent.tools = [PLAY_MOVE_TOOL, RUN_PYTHON_TOOL, SIMULATE_MOVE_TOOL]
    agent.env = object()
    agent.chess_client = object()
    agent.python_sandbox_port = 8000
    agent.last_state = {"game_over": False}
    agent.finished = False
    agent.format_state = Mock(return_value="<chess_state>fresh</chess_state>")
    error = "<chess_error>request failed after a possible move</chess_error>"
    play = Mock(return_value=error if first_fails else '{"game_over": false}')
    run = Mock(return_value=error if first_fails else '{"stdout":"","stderr":"","error":null}')
    simulate = Mock(return_value='{"fen":"hypothetical"}')
    refresh = Mock(return_value={"game_over": False})
    monkeypatch.setattr("assignment.agent.chess_agent._play_move", play)
    monkeypatch.setattr("assignment.agent.chess_agent._run_python", run)
    monkeypatch.setattr("assignment.agent.chess_agent._simulate_move", simulate)
    monkeypatch.setattr("assignment.agent.chess_agent._game_state", refresh)
    calls = [call(names[0], call_id="first"),
             call("simulate_move", call_id="safe"),
             call(names[1], call_id="second")]
    results = agent.execute_tool_calls(calls)
    assert [r["tool_call_id"] for r in results] == ["first", "safe", "second"]
    assert play.call_count + run.call_count == 1
    assert (play.call_count == 1) == (names[0] == "play_move")
    assert results[2]["content"].startswith("<chess_error>")
    assert "Only one" in results[2]["content"]
    simulate.assert_called_once()
    assert refresh.call_count == (1 if names[0] == "run_python" else 0)
