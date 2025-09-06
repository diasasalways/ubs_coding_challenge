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


def _simulate_turn(
    start_square: int,
    first_roll: int,
    is_last_player: bool,
    last_square: int,
    fixed_jumps: Dict[int, int],
    smoke_squares: Set[int],
    mirror_squares: Set[int],
) -> Tuple[int, List[int]]:
    used_rolls: List[int] = [first_roll]
    square = _apply_bounce_forward(start_square, first_roll, last_square)
    # Apply fixed jump if any
    if square in fixed_jumps:
        square = fixed_jumps[square]
    # Resolve any dynamic squares (smoke/mirror), possibly chaining
    while True:
        if square == last_square:
            return square, used_rolls

        if square in smoke_squares:
            # Move backwards by the next roll
            next_roll = 1 if is_last_player else 6
            used_rolls.append(next_roll)
            square = max(1, square - next_roll)
            if square in fixed_jumps:
                square = fixed_jumps[square]
            # continue loop to handle subsequent effects
            continue

        if square in mirror_squares:
            # Move forwards by the next roll
            next_roll = 6 if is_last_player else 1
            used_rolls.append(next_roll)
            square = _apply_bounce_forward(square, next_roll, last_square)
            if square in fixed_jumps:
                square = fixed_jumps[square]
            # continue loop to handle subsequent effects
            continue

        break

    return square, used_rolls


def _choose_roll_for_player(
    start_square: int,
    is_last_player: bool,
    last_square: int,
    fixed_jumps: Dict[int, int],
    smoke_squares: Set[int],
    mirror_squares: Set[int],
) -> Tuple[int, List[int], int]:
    """Return (first_roll_choice, all_rolls_used_this_turn, end_square)."""
    best_first_roll = 1
    best_rolls_used: List[int] = [1]
    best_end_square = start_square

    # Prefer immediate win for last player if possible
    if is_last_player:
        for r in range(1, 7):
            end_sq, rolls_used = _simulate_turn(
                start_square,
                r,
                True,
                last_square,
                fixed_jumps,
                smoke_squares,
                mirror_squares,
            )
            if end_sq == last_square:
                return r, rolls_used, end_sq

    # Evaluate options
    for r in range(1, 7):
        end_sq, rolls_used = _simulate_turn(
            start_square,
            r,
            is_last_player,
            last_square,
            fixed_jumps,
            smoke_squares,
            mirror_squares,
        )

        if is_last_player:
            # Maximize progress, prefer fewer rolls to improve score
            if end_sq > best_end_square or (
                end_sq == best_end_square and len(rolls_used) < len(best_rolls_used)
            ):
                best_first_roll, best_rolls_used, best_end_square = r, rolls_used, end_sq
        else:
            # Avoid letting non-last players win; otherwise minimize progress and prefer fewer rolls
            if end_sq == last_square:
                # Disprefer any result that lets a non-last player win
                continue

            if (
                best_end_square == last_square or
                end_sq < best_end_square or
                (end_sq == best_end_square and len(rolls_used) < len(best_rolls_used))
            ):
                best_first_roll, best_rolls_used, best_end_square = r, rolls_used, end_sq

    return best_first_roll, best_rolls_used, best_end_square


def _solve_slsm(board_size: int, players: int, jumps: List[str]) -> List[int]:
    if board_size <= 0 or players < 2:
        return []
    last_square = board_size
    positions: List[int] = [1 for _ in range(players)]
    fixed_jumps, smoke_squares, mirror_squares = _parse_slsm_jumps(jumps)

    all_rolls: List[int] = []
    current_player = 0
    safety_roll_cap = max(2000, board_size * 10)

    while len(all_rolls) < safety_roll_cap:
        is_last = (current_player == players - 1)
        start_sq = positions[current_player]
        first_roll, rolls_used, end_sq = _choose_roll_for_player(
            start_sq,
            is_last,
            last_square,
            fixed_jumps,
            smoke_squares,
            mirror_squares,
        )

        all_rolls.extend(rolls_used)
        positions[current_player] = end_sq

        if end_sq == last_square:
            break

        current_player = (current_player + 1) % players

    return all_rolls


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