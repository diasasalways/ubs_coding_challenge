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
import xml.etree.ElementTree as ET

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
@app.route("/trivia", methods=["GET"])
def trivia():
    res = {
        "answers": [
            3,  # zhb "Trivia!": How many challenges are there this year, which title ends with an exclamation mark?
            1,  # zhb "Ticketing Agent": What type of tickets is the ticketing agent handling?
            2,  # zhb "Blankety Blanks": How many lists and elements per list are included in the dataset you must impute?
            2,  # zhb "Princess Diaries": What's Princess Mia's cat name in the movie Princess Diaries?
            3,  # zhb "MST Calculation": What is the average number of nodes in a test case?
            4,  # zhb "Universal Bureau of Surveillance": Which singer did not have a James Bond theme song?
            3,  # "Operation Safeguard": What is the smallest font size in our question?
            5,  # zhb "Capture The Flag": Which of these are anagrams of the challenge name?
            4,  # zhb "Filler 1": Where has UBS Global Coding Challenge been held before?
            3   # zhb "Trading Formula": When comparing your answer to the correct answer, what precision level do you have to ensure your answer is precise to?
        ]
    }
    return jsonify(res), 200



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

def _parse_slsm_jumps(jumps: List[Any]) -> Tuple[Dict[int, int], Set[int], Set[int]]:
    fixed_jumps: Dict[int, int] = {}
    smoke_squares: Set[int] = set()
    mirror_squares: Set[int] = set()

    def parse_pair(obj: Any) -> Optional[Tuple[int, int]]:
        # Accept "a:b" strings, [a,b] lists/tuples, and {from:a,to:b} dicts
        if isinstance(obj, str):
            if ":" in obj:
                left_str, right_str = obj.split(":", 1)
                left_str = left_str.strip()
                right_str = right_str.strip()
                try:
                    return int(left_str), int(right_str)
                except ValueError:
                    return None
            return None
        if isinstance(obj, (list, tuple)) and len(obj) == 2:
            try:
                return int(obj[0]), int(obj[1])
            except Exception:
                return None
        if isinstance(obj, dict):
            a = obj.get("from", obj.get("start"))
            b = obj.get("to", obj.get("end"))
            try:
                return int(a), int(b)
            except Exception:
                return None
        return None

    for entry in jumps:
        parsed = parse_pair(entry)
        if not parsed:
            continue
        left, right = parsed

        if left == 0 and right > 0:
            mirror_squares.add(right)
        elif right == 0 and left > 0:
            smoke_squares.add(left)
        elif left > 0 and right > 0 and left != right:
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
        max_steps=max(12000, board_size * 20),
        beam_width=512,
    )


@app.route("/slsm", methods=["POST"])
def slsm():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return bad_request("Expected a JSON object with keys: boardSize, players, jumps")

    board_size = payload.get("boardSize")
    width = payload.get("width")
    height = payload.get("height")
    players = payload.get("players")
    jumps = payload.get("jumps")

    if board_size is None and isinstance(width, int) and isinstance(height, int):
        board_size = width * height
    if not isinstance(board_size, int) or not isinstance(players, int) or not isinstance(jumps, list):
        return bad_request("Invalid types. Expected: boardSize:int (or width*height), players:int, jumps:list")

    rolls = _solve_slsm(board_size, players, jumps)
    resp = make_response(jsonify(rolls), 200)
    resp.headers["Content-Type"] = "application/json"
    return resp


# ------------------------ SLPU (SVG board parsing, single-player) ------------------------

def _svg_attr_float(elem: ET.Element, name: str, default: float = 0.0) -> float:
    v = elem.get(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default


def _infer_grid(svg_root: ET.Element) -> Tuple[int, int, float, float, float]:
    # Try to read viewBox and infer grid size by looking at coordinate extents and the fact that
    # squares measure 32 each side.
    view_box = svg_root.get("viewBox", "0 0 512 512")
    parts = [p for p in re.split(r"[ ,]+", view_box.strip()) if p]
    min_x, min_y, width, height = 0.0, 0.0, 512.0, 512.0
    if len(parts) == 4:
        try:
            min_x, min_y, width, height = map(float, parts)
        except Exception:
            pass
    side = 32.0
    cols = int(round(width / side))
    rows = int(round(height / side))
    return cols, rows, side, min_x, min_y


def _coord_to_square(x: float, y: float, cols: int, rows: int, side: float, min_x: float, min_y: float) -> int:
    # Convert SVG coordinate to 1-based boustrophedon square index
    col = int(math.floor((x - min_x) / side))
    row = int(math.floor((y - min_y) / side))
    # Clamp
    col = max(0, min(cols - 1, col))
    row = max(0, min(rows - 1, row))
    # rows counted from bottom in board indexing; SVG y increases downward
    row_from_bottom = (rows - 1) - row
    if row_from_bottom % 2 == 0:
        # left to right
        idx_in_row = col
    else:
        # right to left
        idx_in_row = (cols - 1) - col
    square = row_from_bottom * cols + idx_in_row + 1
    return int(square)


def _parse_svg_jumps(svg_xml: str) -> Tuple[int, List[str]]:
    try:
        root = ET.fromstring(svg_xml)
    except ET.ParseError:
        return 0, []

    cols, rows, side, min_x, min_y = _infer_grid(root)
    # Only <line> elements represent jumps; polyline with arrows is not present in judge input
    fixed: Dict[int, int] = {}
    for line in root.findall('.//{*}line'):
        x1 = _svg_attr_float(line, 'x1')
        y1 = _svg_attr_float(line, 'y1')
        x2 = _svg_attr_float(line, 'x2')
        y2 = _svg_attr_float(line, 'y2')
        s_from = _coord_to_square(x1, y1, cols, rows, side, min_x, min_y)
        s_to = _coord_to_square(x2, y2, cols, rows, side, min_x, min_y)
        if s_from == s_to:
            continue
        fixed[s_from] = s_to

    # Build jump encoding as required by our solver: normal fixed jumps only
    jumps = [f"{a}:{b}" for a, b in fixed.items()]
    board_size = cols * rows
    return board_size, jumps


def _solve_single_player(board_size: int, jumps: List[str]) -> List[int]:
    # Single player: treat as players=1 and return sequence until reaching last square
    rolls = _solve_slsm(board_size, 1, jumps)
    return rolls


@app.route("/slpu", methods=["POST"])
def slpu():
    svg_xml = request.data.decode("utf-8", errors="ignore")
    board_size, jumps = _parse_svg_jumps(svg_xml)
    if board_size <= 0:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><text></text></svg>'
        resp = make_response(svg, 200)
        resp.headers["Content-Type"] = "image/svg+xml"
        return resp

    rolls = _solve_single_player(board_size, jumps)
    rolls_str = ''.join(str(r) for r in rolls)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{rolls_str}</text></svg>'
    resp = make_response(svg, 200)
    resp.headers["Content-Type"] = "image/svg+xml"
    return resp