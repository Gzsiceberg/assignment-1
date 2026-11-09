"""Provided Part 3 server extension; leaves the pinned Part 1 testbed untouched.

Run this file inside the chess testbed, or locally with chess_app/src on
PYTHONPATH. Simulation reconstructs a position from FEN, never the live game.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from threading import Lock

import chess
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from chess_app.server import MoveRequest, create_app as create_chess_app

class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    fen: str = Field(min_length=1, max_length=200)
    move: str | None = Field(default=None, min_length=4, max_length=5)

def simulate_position(fen: str, move: str | None = None) -> dict:
    """Inspect a position or apply exactly one legal ply, for either color.

    FEN preserves turn, castling, en passant, and move counters, but not a
    repetition history. Repetition-based draws cannot be inferred here.
    """
    try:
        if len(fen.split()) != 6:
            raise ValueError("Expected all six FEN fields.")
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError("The FEN describes an invalid chess position.")
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc

    # Unlike the live game, no repetition history or automatic bot is involved.
    outcome = board.outcome(claim_draw=True)
    applied_move = None
    if move is not None:
        if outcome is not None:
            raise ValueError("Cannot move in a terminal position.")
        try:
            candidate = chess.Move.from_uci(move)
        except ValueError as exc:
            raise ValueError("Move must use UCI notation, including a promotion suffix.") from exc
        if candidate not in board.legal_moves:
            raise ValueError(f"{move} is not legal in the supplied position.")
        board.push(candidate)
        applied_move = candidate.uci()
        outcome = board.outcome(claim_draw=True)

    winner = None if outcome is None or outcome.winner is None else (
        "white" if outcome.winner else "black"
    )
    turn = "white" if board.turn else "black"
    return {
        "fen": board.fen(en_passant="fen"),
        "squares": {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()},
        "turn": turn,
        "legal_moves": sorted(m.uci() for m in board.legal_moves) if outcome is None else [],
        "in_check": board.is_check(),
        "game_over": outcome is not None,
        "winner": winner,
        "result": outcome.result() if outcome else "*",
        "termination": outcome.termination.name.lower() if outcome else None,
        "status": (f"{winner.title()} wins" if winner else "Draw") if outcome else f"{turn.title()} to move",
        "applied_move": applied_move,
        "simulation": True,
        "repetition_history_available": False,
    }

def create_app(game=None) -> FastAPI:
    app = create_chess_app(game)

    # Serialize commits and cache their responses before sending them through
    # the tunnel. A lost response can then be retried without a second move.
    lock = Lock()
    completed = OrderedDict()
    move_route = next(r for r in app.routes if getattr(r, "path", None) == "/api/move")
    reset_route = next(r for r in app.routes if getattr(r, "path", None) == "/api/reset")
    original_move, original_reset = move_route.endpoint, reset_route.endpoint
    app.routes.remove(move_route)
    app.routes.remove(reset_route)

    @app.post("/api/move")
    def play_move(
        request: MoveRequest,
        idempotency_key: str | None = Header(default=None, max_length=128),
    ) -> dict:
        with lock:
            if idempotency_key and idempotency_key in completed:
                previous_move, result = completed[idempotency_key]
                if previous_move != request.move:
                    raise HTTPException(status_code=409, detail="Request ID reused for a different move")
                return result
            result = original_move(request)
            if idempotency_key:
                completed[idempotency_key] = (request.move, result)
                if len(completed) > 1024:
                    completed.popitem(last=False)
            return result

    @app.post("/api/reset")
    def reset_game() -> dict:
        with lock:
            # Retain IDs so delayed retries cannot commit in the new game.
            return original_reset()

    @app.post("/api/simulate")
    def simulate(request: SimulationRequest) -> dict:
        try:
            return simulate_position(request.fen, request.move)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess server with stateless simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)
