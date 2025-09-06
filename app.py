from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from flask_cors import CORS
import random
from collections import defaultdict, deque, Counter
import re, math
from dataclasses import dataclass

app = Flask(__name__)

def bad_request(message: str):
    resp = make_response(jsonify({"error": message}), 400)
    resp.headers["Content-Type"] = "application/json"
    return resp

if __name__ == "__main__":
    # For local development only
    app.run()
    # app.run(host='0.0.0.0', port=3000, debug=False)

@app.route("/")
def root():
    return "OK", 200

@app.route("/The-Ink-Archive", methods=["POST"]) 
def the_ink_archive():
    payload = request.get_json(silent=True)
    if payload is None or not isinstance(payload, list) or len(payload) == 0:
        return bad_request("Expected a JSON array with two items for Part I and Part II.")

    results: List[Dict[str, Any]] = []

    # Part I: detect any profitable loop
    part1 = payload[0] if len(payload) >= 1 else {}
    goods1 = part1.get("goods", []) if isinstance(part1, dict) else []
    rates1 = part1.get("rates", []) if isinstance(part1, dict) else []
    n1, edges1, rate_map1 = _ink_build_graph(goods1 if isinstance(goods1, list) else [], rates1)

    if n1 > 0:
        # For Part I, return the most profitable cycle as well (stable and aligns with samples)
        best_cycle1, best_gain1 = _ink_max_gain_cycle(n1, edges1, rate_map1)
        if best_cycle1:
            can1 = _ink_canonicalize_cycle(best_cycle1)
            path1_names = [goods1[i] for i in can1]
            results.append({"path": path1_names, "gain": best_gain1})
        else:
            results.append({"path": [], "gain": 0.0})
    else:
        results.append({"path": [], "gain": 0.0})

    # Part II: find the maximum gain cycle
    part2 = payload[1] if len(payload) >= 2 else {}
    goods2 = part2.get("goods", []) if isinstance(part2, dict) else []
    rates2 = part2.get("rates", []) if isinstance(part2, dict) else []
    n2, edges2, rate_map2 = _ink_build_graph(goods2 if isinstance(goods2, list) else [], rates2)

    if n2 > 0:
        best_cycle2, best_gain2 = _ink_max_gain_cycle(n2, edges2, rate_map2)
        if best_cycle2:
            can2 = _ink_canonicalize_cycle(best_cycle2)
            path2_names = [goods2[i] for i in can2]
            results.append({"path": path2_names, "gain": best_gain2})
        else:
            results.append({"path": [], "gain": 0.0})
    else:
        results.append({"path": [], "gain": 0.0})

    resp = make_response(jsonify(results), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


# ------------------------ SLSM (Snakes & Ladders, Smoke & Mirrors) ------------------------

def _parse_slsm_jumps(jumps: List[str]) -> Tuple[Dict[int, int], Set[int], Set[int]]:
    fixed_jumps: Dict[int, int] = {}
    smoke_squares: Set[int] = set()
    mirror_squares: Set[int] = set()
    for entry in jumps:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        left_str, right_str = entry.split(":", 1)
        try:
            left = int(left_str)
            right = int(right_str)
        except ValueError:
            continue

        if left == 0 and right > 0:
            # Mirror at 'right'
            mirror_squares.add(right)
        elif right == 0 and left > 0:
            # Smoke at 'left'
            smoke_squares.add(left)
        elif left > 0 and right > 0 and left != right:
            # Fixed jump (snake or ladder)
            fixed_jumps[left] = right
    return fixed_jumps, smoke_squares, mirror_squares


def _apply_bounce_forward(current_square: int, roll_value: int, last_square: int) -> int:
    tentative = current_square + roll_value
    if tentative > last_square:
        overshoot = tentative - last_square
        return last_square - overshoot
    return tentative


@dataclass(frozen=True)
class _PlannerState:
    positions: Tuple[int, ...]
    current_player: int
    # -1: resolve Smoke (backward by next roll), 0: none, +1: resolve Mirror (forward by next roll)
    pending_effect: int


def _resolve_fixed_jumps(square: int, fixed_jumps: Dict[int, int]) -> int:
    # Follow chains if any, defensively cap at a reasonable number
    for _ in range(8):
        nxt = fixed_jumps.get(square)
        if nxt is None:
            return square
        square = nxt
    return square


def _apply_roll(
    state: _PlannerState,
    roll: int,
    last_square: int,
    fixed_jumps: Dict[int, int],
    smoke_squares: Set[int],
    mirror_squares: Set[int],
) -> Tuple[_PlannerState, int]:
    positions = list(state.positions)
    p = state.current_player

    # Determine direction of movement
    if state.pending_effect == -1:
        # Smoke resolution: move backwards by roll
        new_pos = max(1, positions[p] - roll)
        new_pos = _resolve_fixed_jumps(new_pos, fixed_jumps)
        # After resolving, check for another dynamic tile
        if new_pos in smoke_squares:
            next_effect = -1
            next_player = p
        elif new_pos in mirror_squares:
            next_effect = +1
            next_player = p
        else:
            next_effect = 0
            next_player = (p + 1) % len(positions)

    else:
        # Normal or Mirror resolution (both forward with bounce)
        new_pos = _apply_bounce_forward(positions[p], roll, last_square)
        new_pos = _resolve_fixed_jumps(new_pos, fixed_jumps)
        if new_pos == last_square:
            positions[p] = new_pos
            return _PlannerState(tuple(positions), p, 0), p

        if new_pos in smoke_squares:
            next_effect = -1
            next_player = p
        elif new_pos in mirror_squares:
            next_effect = +1
            next_player = p
        else:
            next_effect = 0
            next_player = (p + 1) % len(positions)

    positions[p] = new_pos
    return _PlannerState(tuple(positions), next_player, next_effect), -1


def _heuristic(
    state: _PlannerState,
    last_square: int,
) -> float:
    positions = state.positions
    last_idx = len(positions) - 1
    last_pos = positions[last_idx]
    best_other = max(positions[:last_idx]) if last_idx > 0 else 1
    dist_last = last_square - last_pos
    dist_other = last_square - best_other
    # Larger is better
    return 1000.0 * (dist_other - dist_last) - 5.0 * (state.pending_effect != 0)


def _beam_search_plan(
    board_size: int,
    players: int,
    fixed_jumps: Dict[int, int],
    smoke_squares: Set[int],
    mirror_squares: Set[int],
    max_steps: int = 4000,
    beam_width: int = 128,
) -> List[int]:
    start = _PlannerState(tuple([1] * players), 0, 0)
    last_square = board_size

    # Each entry: (state, rolls_so_far)
    beam: List[Tuple[_PlannerState, List[int]]] = [(start, [])]
    seen_best_depth: Dict[Tuple, int] = { (start.positions, start.current_player, start.pending_effect): 0 }

    for depth in range(max_steps):
        candidates: List[Tuple[_PlannerState, List[int]]] = []
        for state, seq in beam:
            # If any non-last player already finished, drop this branch
            for i in range(players - 1):
                if state.positions[i] == last_square:
                    break
            else:
                # Expand six possible rolls
                for r in (1, 2, 3, 4, 5, 6):
                    next_state, winner = _apply_roll(state, r, last_square, fixed_jumps, smoke_squares, mirror_squares)
                    new_seq = seq + [r]
                    if winner == players - 1:
                        return new_seq
                    # If someone else won, skip
                    someone_else_won = (winner != -1 and winner != players - 1)
                    if someone_else_won:
                        continue
                    key = (next_state.positions, next_state.current_player, next_state.pending_effect)
                    prev_best = seen_best_depth.get(key)
                    if prev_best is None or prev_best > len(new_seq):
                        seen_best_depth[key] = len(new_seq)
                        candidates.append((next_state, new_seq))

        if not candidates:
            break

        # Keep top beam_width by heuristic
        candidates.sort(key=lambda item: _heuristic(item[0], last_square), reverse=True)
        beam = candidates[:beam_width]

    # Fallback: no plan found within limits. Return a safe default of rolling 1s.
    return [1] * min(100, max_steps)


def _solve_slsm(board_size: int, players: int, jumps: List[str]) -> List[int]:
    if board_size <= 0 or players < 2:
        return []
    fixed_jumps, smoke_squares, mirror_squares = _parse_slsm_jumps(jumps)
    # Beam-search for a sequence that makes the last player win first
    return _beam_search_plan(
        board_size,
        players,
        fixed_jumps,
        smoke_squares,
        mirror_squares,
        max_steps=max(4000, board_size * 12),
        beam_width=160,
    )


@app.route("/slsm", methods=["POST"])
def slsm():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Expected a JSON object with keys: boardSize, players, jumps")

    board_size = payload.get("boardSize")
    players = payload.get("players")
    jumps = payload.get("jumps")

    if not isinstance(board_size, int) or not isinstance(players, int) or not isinstance(jumps, list):
        return bad_request("Invalid types. Expected: boardSize:int, players:int, jumps:list[str]")

    # Minimal sanity checks per constraints
    if players < 2 or players > 8:
        return bad_request("players must be between 2 and 8 inclusive")
    if board_size < 64 or board_size > 400:
        return bad_request("boardSize must be between 64 and 400 inclusive")

    rolls = _solve_slsm(board_size, players, jumps)
    resp = make_response(jsonify(rolls), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp