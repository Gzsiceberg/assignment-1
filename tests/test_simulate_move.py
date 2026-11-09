"""Offline simulation tool tests; mock responses do not test the chess engine."""

import copy
import json

import httpx
import pytest

from assignment.agent import ChessAgent
from assignment.agent.chess_tools import _simulate_move
from assignment.agent.tools import SIMULATE_MOVE_TOOL


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_WHITE_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
AFTER_BLACK_FEN = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
INITIAL_STATE = {
    "fen": START_FEN,
    "squares": {"e1": "K", "e2": "P", "e7": "p", "e8": "k"},
    "turn": "white",
    "legal_moves": ["e2e4", "d2d4"],
    "game_over": False,
    "in_check": False,
    "status": "Your turn",
    "history": [],
}
WHITE_RESULT = {
    **INITIAL_STATE,
    "fen": AFTER_WHITE_FEN,
    "squares": {"e1": "K", "e4": "P", "e7": "p", "e8": "k"},
    "turn": "black",
    "legal_moves": ["e7e5"],
}
BLACK_RESULT = {
    **INITIAL_STATE,
    "fen": AFTER_BLACK_FEN,
    "squares": {"e1": "K", "e4": "P", "e5": "p", "e8": "k"},
    "legal_moves": ["g1f3"],
}


@pytest.fixture
def simulation_client():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path in ("/api/reset", "/api/state"):
            return httpx.Response(200, json=INITIAL_STATE)
        if request.url.path == "/api/simulate" and request.method == "POST":
            args = json.loads(request.content)
            position = args.get("fen"), args.get("move")
            states = {
                (START_FEN, None): INITIAL_STATE,
                (START_FEN, "e2e4"): WHITE_RESULT,
                (AFTER_WHITE_FEN, "e7e5"): BLACK_RESULT,
            }
            if position in states:
                return httpx.Response(200, json=states[position])
            return httpx.Response(400, json={"detail": "invalid position or illegal move"})
        return httpx.Response(404, json={"detail": "not found"})

    with httpx.Client(
        base_url="http://chess.test", transport=httpx.MockTransport(handler)
    ) as client:
        client.requests = requests
        yield client


@pytest.fixture
def simulation_agent(monkeypatch, simulation_client):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.test/v1")

    class Environment:
        cwd = "/testbed"
        server_url = "http://chess.test"
        system, release, version, machine = "Linux", "6.1", "#1", "x86_64"

    return ChessAgent(
        Environment(), model="test-model", http_client=simulation_client,
        auto_stop_environment=False,
    )


def assert_chess_error(result):
    assert isinstance(result, str)
    assert result.startswith("<chess_error>")
    assert result.endswith("</chess_error>")
    assert result.removeprefix("<chess_error>").removesuffix("</chess_error>").strip()


def test_simulate_move_schema_and_registration(simulation_agent):
    assert SIMULATE_MOVE_TOOL["type"] == "function"
    function = SIMULATE_MOVE_TOOL["function"]
    assert function["name"] == "simulate_move"
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert parameters["properties"]["fen"]["type"] == "string"
    assert set(parameters["properties"]) == {"fen", "move"}
    assert "fen" in parameters["required"]
    assert parameters["additionalProperties"] is False
    assert SIMULATE_MOVE_TOOL in simulation_agent.tools


@pytest.mark.parametrize("args, expected", [
    ({"fen": START_FEN}, INITIAL_STATE),
    ({"fen": START_FEN, "move": None}, INITIAL_STATE),
    ({"fen": START_FEN, "move": "e2e4"}, WHITE_RESULT),
    ({"fen": AFTER_WHITE_FEN, "move": "e7e5"}, BLACK_RESULT),
])
def test_simulate_move_posts_request_and_returns_json(simulation_client, args, expected):
    result = _simulate_move(simulation_client, json.dumps(args))
    assert json.loads(result) == expected
    request = simulation_client.requests[-1]
    assert request.method == "POST"
    assert request.url.path == "/api/simulate"
    # An omitted optional move and an explicit null have the same meaning.
    sent = json.loads(request.content)
    assert sent["fen"] == args["fen"]
    assert sent.get("move") == args.get("move")
    assert set(sent) <= {"fen", "move"}


@pytest.mark.parametrize("arguments", [
    "{not json", "[]", "null", '"fen"', "{}",
    json.dumps({"fen": 123}),
    json.dumps({"fen": None}),
    json.dumps({"fen": START_FEN, "move": 123}),
    json.dumps({"fen": START_FEN, "move": []}),
])
def test_simulate_move_invalid_arguments_are_recoverable(simulation_client, arguments):
    assert_chess_error(_simulate_move(simulation_client, arguments))
    assert simulation_client.requests == []


@pytest.mark.parametrize("args", [
    {"fen": "invalid-fen"},
    {"fen": START_FEN, "move": "e2e5"},
])
def test_simulate_move_server_rejections_are_recoverable(simulation_client, args):
    assert_chess_error(_simulate_move(simulation_client, json.dumps(args)))
    assert len(simulation_client.requests) == 1


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_simulate_move_transport_failures_are_recoverable(error_type):
    def handler(request):
        raise error_type("connection failed", request=request)

    with httpx.Client(
        base_url="http://chess.test", transport=httpx.MockTransport(handler)
    ) as client:
        assert_chess_error(_simulate_move(client, json.dumps({"fen": START_FEN})))


def test_simulate_move_dispatch_preserves_live_state(simulation_agent, simulation_client):
    before = copy.deepcopy(simulation_agent.last_state)
    finished = simulation_agent.finished
    calls = [
        {
            "id": f"simulate_{index}",
            "type": "function",
            "function": {"name": "simulate_move", "arguments": json.dumps(args)},
        }
        for index, args in enumerate([
            {"fen": START_FEN, "move": "e2e4"},
            {"fen": AFTER_WHITE_FEN, "move": "e7e5"},
        ])
    ]
    messages = simulation_agent.execute_tool_calls(calls)
    assert len(messages) == 2
    for message, call, expected in zip(messages, calls, [WHITE_RESULT, BLACK_RESULT]):
        assert message["role"] == "tool"
        assert message["tool_call_id"] == call["id"]
        assert json.loads(message["content"]) == expected
    assert simulation_agent.last_state == before
    assert simulation_agent.finished is finished
    assert simulation_client.get("/api/state").json() == before
    assert [r.url.path for r in simulation_client.requests] == [
        "/api/reset", "/api/simulate", "/api/simulate", "/api/state",
    ]
