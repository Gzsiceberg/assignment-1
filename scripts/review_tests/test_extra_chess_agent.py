"""Fast, offline tests for the student-implemented chess tool."""

import json

import httpx
import pytest

from assignment.agent import ChessAgent
from assignment.agent.chess_agent import format_chess_state
from assignment.agent.tools import INVOKE_SKILL_TOOL, PLAY_MOVE_TOOL
from assignment.agent.chess_tools import _invoke_skill


INITIAL_STATE = {
    "status": "White to move",
    "turn": "white",
    "in_check": False,
    "game_over": False,
    "fen": "initial-fen",
    "squares": {"e1": "K", "e2": "P", "e7": "p", "e8": "k"},
    "history": [],
    "legal_moves": ["e2e4", "d2d4"],
    "human_move": None,
    "engine_move": None,
}

UPDATED_STATE = {
    **INITIAL_STATE,
    "fen": "updated-fen",
    "history": [
        {"ply": 1, "san": "e4", "uci": "e2e4"},
        {"ply": 2, "san": "e5", "uci": "e7e5"},
    ],
    "legal_moves": ["g1f3", "f1c4"],
    "human_move": "e2e4",
    "engine_move": "e7e5",
}


class FakeChessEnvironment:
    cwd = "/testbed"
    server_url = "http://chess.test"
    system, release, version, machine = "Linux", "6.1", "#1", "x86_64"

    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


@pytest.fixture
def chess_client():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/reset" and request.method == "POST":
            return httpx.Response(200, json=INITIAL_STATE)
        if request.url.path == "/api/state" and request.method == "GET":
            return httpx.Response(200, json=INITIAL_STATE)
        if request.url.path == "/api/move" and request.method == "POST":
            if json.loads(request.content) == {"move": "e2e4"}:
                return httpx.Response(200, json=UPDATED_STATE)
            return httpx.Response(400, json={"detail": "illegal move"})
        return httpx.Response(404, json={"detail": "not found"})

    client = httpx.Client(
        base_url="http://chess.test",
        transport=httpx.MockTransport(handler),
    )
    client.requests = requests
    yield client
    client.close()


@pytest.fixture
def make_agent(monkeypatch, chess_client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    return ChessAgent(
        FakeChessEnvironment(),
        model="test-model",
        http_client=chess_client,
        auto_stop_environment=False,
    )


def tool_call(move: str, call_id: str = "move_1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "play_move",
            "arguments": json.dumps({"move": move}),
        },
    }


def test_play_move_schema_is_strict_and_registered(make_agent):
    function = PLAY_MOVE_TOOL["function"]
    parameters = function["parameters"]

    assert PLAY_MOVE_TOOL["type"] == "function"
    assert function["name"] == "play_move"
    assert function["strict"] is True
    assert parameters["properties"]["move"]["type"] == "string"
    assert parameters["required"] == ["move"]
    assert parameters["additionalProperties"] is False
    assert PLAY_MOVE_TOOL in make_agent.tools


def test_initial_prompt_contains_board_and_legal_moves(make_agent):
    prompt = json.dumps(make_agent.build_prompt())
    assert "a b c d e f g h" in prompt
    assert "legal_moves" in prompt
    assert "e2e4" in prompt


def test_board_only_formatter_omits_legal_moves_without_losing_position():
    observation = format_chess_state(INITIAL_STATE, include_legal_moves=False)

    assert "a b c d e f g h" in observation
    assert "fen: initial-fen" in observation
    assert "legal_moves:" not in observation
    assert "e2e4" not in observation


def test_play_move_posts_request_and_returns_new_state(make_agent, chess_client):
    messages = make_agent.execute_tool_calls([tool_call("e2e4")])

    move_request = chess_client.requests[-1]
    assert move_request.method == "POST"
    assert move_request.url.path == "/api/move"
    assert json.loads(move_request.content) == {"move": "e2e4"}
    assert messages[0]["tool_call_id"] == "move_1"
    assert "engine_move: e7e5" in messages[0]["content"]
    assert "legal_moves: g1f3 f1c4" in messages[0]["content"]
    assert make_agent.last_state == UPDATED_STATE
    assert make_agent.finished is False


def test_rejected_and_malformed_moves_are_recoverable(make_agent):
    rejected = make_agent.execute_tool_calls([tool_call("e2e5")])
    malformed_call = tool_call("e2e4", "move_2")
    malformed_call["function"]["arguments"] = "{not json"
    malformed = make_agent.execute_tool_calls([malformed_call])

    assert "<chess_error>" in rejected[0]["content"]
    assert "<chess_error>" in malformed[0]["content"]
    assert make_agent.finished is False


def test_invoke_skill_returns_full_content():
    content = "---\nname: select-move\n---\nFull instructions.\n"
    skills = {"select-move": {"metadata": "catalog entry", "content": content}}
    assert _invoke_skill(skills, '{"name": "select-move"}') == content


@pytest.mark.parametrize("arguments", [
    "{bad", "[]", "null", "{}", '{"name": 123}',
    '{"name": "missing"}', '{"name": "select-move", "extra": true}',
])
def test_invoke_skill_errors_are_recoverable(arguments):
    result = _invoke_skill({}, arguments)
    assert result.startswith("<chess_error>")
    assert result.endswith("</chess_error>")


def test_chess_skill_registration_and_dispatch(monkeypatch, chess_client, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")
    skill_dir = tmp_path / "select-move"
    skill_dir.mkdir()
    content = (
        "---\nname: select-move\ndescription: Search before committing.\n---\n"
        "Use run_python to simulate candidates, then play one move.\n"
    )
    (skill_dir / "SKILL.md").write_text(content)
    agent = ChessAgent(
        FakeChessEnvironment(), model="test-model", http_client=chess_client,
        skills_path=str(tmp_path), auto_stop_environment=False,
    )
    assert agent.tools.count(INVOKE_SKILL_TOOL) == 1
    before_requests = len(chess_client.requests)
    before_state = json.loads(json.dumps(agent.last_state))
    messages = agent.execute_tool_calls([{
        "id": "skill_1", "type": "function",
        "function": {
            "name": "invoke_skill", "arguments": '{"name": "select-move"}',
        },
    }])
    assert messages == [{
        "role": "tool", "tool_call_id": "skill_1", "content": content,
    }]
    assert len(chess_client.requests) == before_requests
    assert agent.last_state == before_state
    assert agent.finished is False


def test_invoke_skill_not_registered_without_skills(make_agent):
    assert INVOKE_SKILL_TOOL not in make_agent.tools
    messages = make_agent.execute_tool_calls([{
        "id": "skill_1", "type": "function",
        "function": {"name": "invoke_skill", "arguments": '{"name": "missing"}'},
    }])
    assert messages[0]["tool_call_id"] == "skill_1"
    assert messages[0]["content"].startswith("<chess_error>Unknown tool:")
