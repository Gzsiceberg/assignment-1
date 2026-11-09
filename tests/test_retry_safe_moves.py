"""Run with python-chess, FastAPI, uvicorn and chess_app/src on PYTHONPATH."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import httpx
import pytest

pytest.importorskip("chess")
pytest.importorskip("fastapi")
pytest.importorskip("chess_app.server")
from fastapi.testclient import TestClient

from assignment.chess_server import create_app
from assignment.agent.chess_tools import _play_move


def game():
    return Mock(play_human_move=Mock(return_value={"game_over": False, "fen": "after"}))


def test_response_lost_after_commit_does_not_play_twice():
    board = game()
    with TestClient(create_app(board)) as server:
        attempts = []

        def handler(request):
            attempts.append(request.headers["Idempotency-Key"])
            response = server.post("/api/move", content=request.content, headers=dict(request.headers))
            if len(attempts) == 1:
                raise httpx.RemoteProtocolError("response lost after commit")
            return httpx.Response(response.status_code, json=response.json())

        with httpx.Client(base_url="http://chess", transport=httpx.MockTransport(handler)) as client:
            assert '"fen": "after"' in _play_move(client, '{"move": "e2e4"}')
    assert len(attempts) == 2
    assert len(set(attempts)) == 1
    board.play_human_move.assert_called_once_with("e2e4")


def test_concurrent_duplicates_and_conflicting_payload():
    board = game()
    with TestClient(create_app(board)) as client:
        def submit(_):
            return client.post("/api/move", json={"move": "e2e4"}, headers={"Idempotency-Key": "same"})
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert all(r.status_code == 200 for r in pool.map(submit, range(4)))
        assert client.post("/api/move", json={"move": "d2d4"}, headers={"Idempotency-Key": "same"}).status_code == 409
        # A new action still commits normally.
        assert client.post("/api/move", json={"move": "d2d4"}, headers={"Idempotency-Key": "new"}).status_code == 200
    assert board.play_human_move.call_count == 2
