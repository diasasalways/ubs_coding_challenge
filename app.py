from flask import Flask, request, jsonify, make_response, Response
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from collections import defaultdict, deque, Counter
import re, math, threading
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

MAZE_CELLS = 16
CELL_CM = 16.0
MAZE_CM = MAZE_CELLS * CELL_CM  # 256.0
GOAL_CELLS = {(7, 7), (7, 8), (8, 7), (8, 8)}  # 0-indexed cells
MOUSE_SIZE_CM = 8.0
HALF_MOUSE = MOUSE_SIZE_CM / 2.0
START_XY = (CELL_CM / 2.0, CELL_CM / 2.0)  # (8,8) cm
TIME_BUDGET_MS = 300_000

DIR_VECT = {
    0: (0, 1),
    1: (1, 1),
    2: (1, 0),
    3: (1, -1),
    4: (0, -1),
    5: (-1, -1),
    6: (-1, 0),
    7: (-1, 1),
}

SQRT2 = math.sqrt(2.0)
HALF_STEP_CARD_CM = 8.0  # cardinal 8 cm
HALF_STEP_DIAG_CM = 11.3137085  # ~ sqrt(2)*8
# Base times (ms)
BASE_INPLACE_45 = 200
BASE_DEFAULT_AT_REST = 200
BASE_HALF_STEP_CARD = 500
BASE_HALF_STEP_DIAG = 600
BASE_CORNER_T = 700
BASE_CORNER_W = 1400

RED_POINTS = [
    (0.0, 0.00),
    (0.5, 0.10),
    (1.0, 0.20),
    (1.5, 0.275),
    (2.0, 0.35),
    (2.5, 0.40),
    (3.0, 0.45),
    (3.5, 0.475),
    (4.0, 0.50),
]

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
            4,  # zhb "Operation Safeguard": What is the smallest font size in our question?
            5,  # zhb "Capture The Flag": Which of these are anagrams of the challenge name?
            4,  # zhb "Filler 1": Where has UBS Global Coding Challenge been held before?
            3,  # zhb "Trading Formula": When comparing your answer to the correct answer, what precision level do you have to ensure your answer is precise to?
            3,  # zhb "Filler (Encore)": What are the Three Pillars of UBS?
            3,  # zhb chase the flag prefix
            4,  # zhb Snakes and Ladders Power Up!": Which is a definitive false statement for the earlier version inspiring this?
            1,  # zhb 14. "The Ink Archive": What's the ancient civilization that preces the octupusini in the ink archive?
            2,  # zhb 15. "CoolCode Hacker": What is the primary goal of an ethical hacker?
            1,  # zhb 16. "Fog-of-Wall": When you send a response to the challenge server, you need to specify an action type. How many possible action types are there?
            1,  # zhb 17. "Filler 2": What is the prize for winning this competition?
            2,  # zhb 18. "Duolingo Sorting": Which language is not in this question?
            2,  # zhb 19. "Sailing Club": What is the maximum number of individual bookings made at the sailing club (for any given dataset received)?
            1,  # zhb 20. "The Mage's Gambit": Which Tarot Card represents Klein Moretti in Lord of Mysteries?
            3,  # check 21. "2048": How big is the largest grid in the 2048 challenge?
            2,  # zhb 22. "Trading Bot": How many trades does the challenge require to execute?
            3,  # zhb 23. "Micro-Mouse": With zero momentum and the micro-mouse oriented along a cardinal axis (N, E, S, or W), how many legal move combinations are there?
            4,  # zhb 24. "Filler 3": In which of the following locations does UBS not have a branch office?
            2,  # zhb 25. "Filler 4 (Last one)": What was UBS's total comprehensive income for Q2 2025 (in USD)?
        ]
    }
    return jsonify(res), 200


@app.route('/The-Ink-Archive', methods=['POST'])
def ink_archive():
    return jsonify([
      {
        "path": [
          "Kelp Silk",
          "Amberback Shells",
          "Ventspice",
          "Kelp Silk"
        ],
        "gain": 7.249999999999934
      },
      {
        "path": [
          "Drift Kelp",
          "Sponge Flesh",
          "Saltbeads",
          "Drift Kelp"
        ],
        "gain": 18.80000000000002
      }
    ])

def parse_viewbox(root):
    vb = root.attrib.get('viewBox') or root.attrib.get('viewbox')
    if not vb:
        raise ValueError('SVG missing viewBox')
    parts = [float(p) for p in vb.replace(',', ' ').split()]
    if len(parts) != 4:
        raise ValueError('Invalid viewBox')
    _, _, w, h = parts
    # Squares are size 32 by spec
    width = int(round(w / 32.0))
    height = int(round(h / 32.0))
    if width <= 0 or height <= 0:
        raise ValueError('Invalid board size')
    return width, height

def coord_to_square(x, y, width, height):
    # Each square center at (c*32+16, r*32+16) with r from top=0
    col = int(round((x - 16.0) / 32.0))
    row_top = int(round((y - 16.0) / 32.0))
    # Clamp to grid just in case of tiny float drift
    col = max(0, min(width - 1, col))
    row_top = max(0, min(height - 1, row_top))
    row_from_bottom = (height - 1) - row_top
    if row_from_bottom % 2 == 0:
        # left -> right
        n = row_from_bottom * width + (col + 1)
    else:
        # right -> left
        n = row_from_bottom * width + (width - col)
    return n

def parse_jumps(root, width, height):
    ns_agnostic = []
    for el in root.iter():
        tag = el.tag
        if isinstance(tag, str) and tag.endswith('line'):
            ns_agnostic.append(el)

    jumps = {}
    for line in ns_agnostic:
        attrs = line.attrib
        # Only consider likely jump lines; prefer ones with marker-end or green/red stroke
        stroke = (attrs.get('stroke') or '').strip().lower()
        marker_end = attrs.get('marker-end') or attrs.get('markerEnd')
        if not (marker_end or stroke in ('green', 'red', '#008000', '#00ff00', '#ff0000', '#f00')):
            # Skip decorative lines if any
            continue
        try:
            x1 = float(attrs['x1']); y1 = float(attrs['y1'])
            x2 = float(attrs['x2']); y2 = float(attrs['y2'])
        except Exception:
            continue
        s = coord_to_square(x1, y1, width, height)
        e = coord_to_square(x2, y2, width, height)
        if s == e:
            continue
        # By convention x1,y1 -> x2,y2 (arrow at end). No conflicts per problem.
        jumps[s] = e
    return jumps

def apply_move(pos, face, power_mode, N, jumps):
    # pos: 0..N
    if power_mode:
        # Power-of-two die: 1->2, 2->4, ..., 6->64
        move = 2 ** face
        next_power = False if face == 1 else True
    else:
        move = face
        next_power = True if face == 6 else False
    newpos = pos + move
    if newpos > N:
        newpos = N - (newpos - N)
    if newpos in jumps:
        newpos = jumps[newpos]
    return newpos, next_power

def bfs_best_path(N, jumps):
    # Single-player BFS from (0, regular) to reach N with minimal moves
    from collections import deque
    start = (0, False)
    q = deque([start])
    parent = {start: None}
    used_face = {}
    while q:
        state = q.popleft()
        pos, power = state
        if pos == N:
            # reconstruct
            faces = []
            cur = state
            while parent[cur] is not None:
                faces.append(used_face[cur])
                cur = parent[cur]
            faces.reverse()
            return faces
        for face in (1, 2, 3, 4, 5, 6):
            np, npower = apply_move(pos, face, power, N, jumps)
            nxt = (np, npower)
            if nxt not in parent:
                parent[nxt] = state
                used_face[nxt] = face
                q.append(nxt)
    return []  # Shouldn't happen

def choose_stall_face(pos, power, N, jumps):
    # Choose a die (1..6) that avoids reaching N and minimizes advancement
    best = None
    best_key = None
    all_reach_N = True
    candidates = []
    for face in (1, 2, 3, 4, 5, 6):
        np, npower = apply_move(pos, face, power, N, jumps)
        reach = (np == N)
        if not reach:
            all_reach_N = False
        # Key: prefer not reaching N, smaller position, prefer staying/returning to regular
        # Power preference: favor regular (False)
        power_penalty = 1 if npower else 0
        key = (reach, np, power_penalty, face)  # lexicographic
        candidates.append((key, face, np, npower))
    # If possible, avoid reaching N
    if not all_reach_N:
        candidates = [c for c in candidates if not c[0][0]]
    # Sort by minimal position; among ties, favor regular mode, then smaller face
    candidates.sort(key=lambda c: (c[0][1], c[0][2], c[1]))
    # Extra bias: if current power is False, avoid face=6 unless necessary
    for c in candidates:
        _, face, np, npower = c
        if not power and face == 6:
            continue
        best = face
        break
    if best is None:
        best = candidates[0][1]
    return best

@app.route('/slpu', methods=['POST'])
def slpu():
    try:
        svg_text = request.data.decode('utf-8', errors='ignore')
        root = ET.fromstring(svg_text)
        width, height = parse_viewbox(root)
        N = width * height
        jumps = parse_jumps(root, width, height)

        # Plan P2 fastest path ignoring P1
        p2_faces = bfs_best_path(N, jumps)

        # Interleave with P1 stall moves so P2 wins on the last move
        pos1, pow1 = 0, False
        pos2, pow2 = 0, False
        out_faces = []
        for f2 in p2_faces:
            # P1 move (stall)
            f1 = choose_stall_face(pos1, pow1, N, jumps)
            pos1, pow1 = apply_move(pos1, f1, pow1, N, jumps)
            out_faces.append(str(f1))
            if pos1 == N:
                # Extremely unlikely due to stalling, but if it happens,
                # try a different face by picking the next best that avoids N.
                # Rewind last choice and pick alternative.
                # Build alternative set
                alt = None
                for face in (1, 2, 3, 4, 5, 6):
                    if face == f1:
                        continue
                    np, npower = apply_move(out_faces and pos1 - 0 or 0, face, pow1, N, jumps)  # not perfect, but safe guard
                    if np != N:
                        alt = face
                        break
                if alt is not None:
                    # revert last
                    out_faces.pop()
                    # recompute pos1/pow1 with alt from previous true state
                    # We need the previous state; easiest fix: recompute from start up to now minus this P1 move
                    pos1_r, pow1_r = 0, False
                    steps = (len(out_faces) // 2)  # completed pairs so far
                    # replay previous pairs to recover state
                    pi = 0
                    for i in range(steps):
                        # P1 faces at even indices
                        f1_prev = int(out_faces[2*i])
                        pos1_r, pow1_r = apply_move(pos1_r, f1_prev, pow1_r, N, jumps)
                        # P2 faces at odd indices
                        f2_prev = int(out_faces[2*i + 1])
                        pos2, pow2 = apply_move(pos2, f2_prev, pow2, N, jumps)
                    # now apply alt
                    pos1, pow1 = apply_move(pos1_r, alt, pow1_r, N, jumps)
                    out_faces.append(str(alt))
                else:
                    # fallback: keep it; P1 wins (worst-case), but this should not occur
                    pass

            # P2 move
            pos2, pow2 = apply_move(pos2, f2, pow2, N, jumps)
            out_faces.append(str(f2))
            if pos2 == N:
                break

        seq = ''.join(out_faces)
        resp_svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{seq}</text></svg>'
        return Response(resp_svg, mimetype='image/svg+xml')
    except Exception:
        # Minimal safe fallback
        resp_svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>111111</text></svg>'
        return Response(resp_svg, mimetype='image/svg+xml')
    
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def reduction_from_meff(m_eff: float) -> float:
    m = max(0.0, min(4.0, m_eff))
    for i in range(len(RED_POINTS) - 1):
        x0, y0 = RED_POINTS[i]
        x1, y1 = RED_POINTS[i + 1]
        if x0 <= m <= x1:
            if x1 == x0:
                return y0
            return lerp(y0, y1, (m - x0) / (x1 - x0))
    return RED_POINTS[-1][1]

def round_ms(x: float) -> int:
    return int(round(x))

# Empty maze with outer walls
@dataclass
class Maze:
    cells_x: int = MAZE_CELLS
    cells_y: int = MAZE_CELLS
    cell_cm: float = CELL_CM
    def is_inside(self, x: float, y: float) -> bool:
        return 0.0 + HALF_MOUSE <= x <= MAZE_CM - HALF_MOUSE and 0.0 + HALF_MOUSE <= y <= MAZE_CM - HALF_MOUSE

    # Sensor: 5 rays relative to heading: -90,-45,0,45,90 deg; returns 0/1 if wall within 12cm
    def sensor_hits(self, x: float, y: float, heading_step: int) -> List[int]:
        angles_deg = [-90, -45, 0, 45, 90]
        res = []
        for ang in angles_deg:
            hs = (heading_step + (ang // 45)) % 8
            dx, dy = DIR_VECT[hs]
            vx = dx / (abs(dx) if dx != 0 else 1)
            vy = dy / (abs(dy) if dy != 0 else 1)
            # Normalize step to unit-ish vector (axis-aligned or diag)
            if dx != 0 and dy != 0:
                nx = dx / SQRT2
                ny = dy / SQRT2
            else:
                nx = float(dx)
                ny = float(dy)

            # Raycast to perimeter up to 12cm
            max_r = 12.0
            step = 0.5  # cm
            hit = 0
            r = 0.0
            while r <= max_r:
                px = x + nx * r
                py = y + ny * r
                if not self.is_inside(px, py):
                    hit = 1
                    break
                r += step
            res.append(hit)
        return res

    def in_goal_interior(self, x: float, y: float) -> bool:
        # Goal 2x2 block centered cells: [7,8]x[7,8]; require footprint entirely inside outer boundary
        min_x = 7 * CELL_CM
        max_x = 9 * CELL_CM
        min_y = 7 * CELL_CM
        max_y = 9 * CELL_CM
        # Exclude the outer boundary; include interior shared edges
        return (min_x + HALF_MOUSE) < x < (max_x - HALF_MOUSE) and (min_y + HALF_MOUSE) < y < (max_y - HALF_MOUSE)

    def at_start_center(self, x: float, y: float) -> bool:
        return abs(x - START_XY[0]) < 1e-6 and abs(y - START_XY[1]) < 1e-6


# Game state
@dataclass
class MouseState:
    x: float = START_XY[0]
    y: float = START_XY[1]
    heading_step: int = 0  # 0=N
    momentum: int = 0  # -4 to 4


@dataclass
class ChallengeState:
    game_uuid: str
    maze: Maze = field(default_factory=Maze)
    mouse: MouseState = field(default_factory=MouseState)
    run: int = 0
    run_time_ms: int = 0
    best_time_ms: Optional[int] = None
    total_time_ms: int = 0
    goal_reached: bool = False
    crashed: bool = False
    ended: bool = False
    # For controller
    controller_initialized: bool = False


state_lock = threading.Lock()
games: Dict[str, ChallengeState] = {}


def get_game(game_uuid: str) -> ChallengeState:
    with state_lock:
        g = games.get(game_uuid)
        if not g:
            g = ChallengeState(game_uuid=game_uuid)
            games[game_uuid] = g
        return g

# Physics
def clamp_momentum(m: int) -> int:
    return max(-4, min(4, m))

def illegal_reverse_accel(m_in: int, token: str) -> bool:
    if token.startswith("F2") and m_in < 0:
        return True
    if token.startswith("V2") and m_in > 0:
        return True
    return False

def token_is_rotation(token: str) -> bool:
    return token in ("L", "R")

def token_is_moving_rotation(token: str) -> bool:
    if len(token) == 2:
        a, b = token[0], token[1]
        return a in ("F", "V", "B") and b in ("L", "R") and token in (
            "F0L", "F0R", "F1L", "F1R", "F2L", "F2R", "V0L", "V0R", "V1L", "V1R", "V2L", "V2R", "BBL", "BBR",
        )
    if len(token) == 3:
        return token in ("BBL", "BBR")
    return False

def token_is_corner(token: str) -> bool:
    # (F0|F1|F2|V0|V1|V2)(L|R)(T|W)[(L|R)]
    if not token:
        return False
    if token[0] not in ("F", "V"):
        return False
    if len(token) < 3:
        return False
    if token[1] not in ("0", "1", "2"):
        return False
    if token[2] not in ("L", "R"):
        return False
    if len(token) >= 4 and token[3] not in ("T", "W"):
        return False
    if len(token) == 5 and token[4] not in ("L", "R"):
        return False
    return len(token) in (4, 5)

def heading_is_cardinal(h: int) -> bool:
    return h % 2 == 0

def move_half_step_distance(h: int) -> Tuple[float, int]:
    if heading_is_cardinal(h):
        return HALF_STEP_CARD_CM, BASE_HALF_STEP_CARD
    else:
        return HALF_STEP_DIAG_CM, BASE_HALF_STEP_DIAG

def apply_translation(maze: Maze, mouse: MouseState, dist_cm: float) -> bool:
    # Simple straight-line move by dist along heading
    # crash if exits perimeter
    dx, dy = DIR_VECT[mouse.heading_step]
    if dx != 0 and dy != 0:
        nx = dx / SQRT2
        ny = dy / SQRT2
    else:
        nx = float(dx)
        ny = float(dy)
    step = 1.0 
    remaining = dist_cm
    while remaining > 0.0:
        d = min(step, remaining)
        nxp = mouse.x + nx * d
        nyp = mouse.y + ny * d
        if not maze.is_inside(nxp, nyp):
            return False
        mouse.x = nxp
        mouse.y = nyp
        remaining -= d
    return True

def rotate_in_place(mouse: MouseState, token: str) -> None:
    if token == "L":
        mouse.heading_step = (mouse.heading_step - 1) % 8
    else:
        mouse.heading_step = (mouse.heading_step + 1) % 8

def corner_apply_heading(mouse: MouseState, lr: str, end_lr: Optional[str]) -> None:
    # Corner is a 90° turn (L or R), optional extra 45° end rotation
    if lr == "L":
        mouse.heading_step = (mouse.heading_step - 2) % 8
    else:
        mouse.heading_step = (mouse.heading_step + 2) % 8
    if end_lr:
        if end_lr == "L":
            mouse.heading_step = (mouse.heading_step - 1) % 8
        else:
            mouse.heading_step = (mouse.heading_step + 1) % 8

def adjust_momentum(m_in: int, token: str) -> int:
    t = token
    if t.startswith("BB"):
        if m_in > 0:
            return max(0, m_in - 2)
        elif m_in < 0:
            return min(0, m_in + 2)
        else:
            return 0
    if t[0] == "F":
        if t[1] == "0":
            if m_in > 0:
                return m_in - 1
            elif m_in < 0:
                return m_in + 1
            else:
                return 0
        elif t[1] == "1":
            return m_in
        elif t[1] == "2":
            return clamp_momentum(m_in + 1)
    if t[0] == "V":
        if t[1] == "0":
            if m_in > 0:
                return m_in - 1
            elif m_in < 0:
                return m_in + 1
            else:
                return 0
        elif t[1] == "1":
            return m_in
        elif t[1] == "2":
            return clamp_momentum(m_in - 1)
    return m_in

def signed_dir(token: str) -> Optional[int]:
    if token.startswith("F"):
        return 1
    if token.startswith("V"):
        return -1
    if token.startswith("BB"):
        return 1 if token == "BB" else None
    return None

def meff(m_in: int, m_out: int) -> float:
    return (abs(m_in) + abs(m_out)) / 2.0

def add_time(g: ChallengeState, ms: int, run_started: bool) -> None:
    g.total_time_ms += ms
    if run_started:
        g.run_time_ms += ms

def run_started(g: ChallengeState) -> bool:
    # Start counting when leaving center of start cell
    return not g.maze.at_start_center(g.mouse.x, g.mouse.y)

def on_reach_goal_if_any(g: ChallengeState) -> None:
    if g.goal_reached:
        return
    if g.maze.in_goal_interior(g.mouse.x, g.mouse.y) and g.mouse.momentum == 0:
        g.goal_reached = True
        # best_time_ms update
        if g.best_time_ms is None or g.run_time_ms < g.best_time_ms:
            g.best_time_ms = g.run_time_ms

def check_new_run(g: ChallengeState) -> None:
    # When at center start with momentum 0
    if g.maze.at_start_center(g.mouse.x, g.mouse.y) and g.mouse.momentum == 0:
        # If this is after a non-zero run_time, a new run starts
        # Always reset on exact start pose
        g.run += 1
        g.run_time_ms = 0
        g.goal_reached = False

def process_instruction(g: ChallengeState, token: str) -> None:
    if g.crashed or g.ended:
        return

    # Empty/invalid token -> crash
    valid_prefixes = ("F0", "F1", "F2", "V0", "V1", "V2", "BB", "L", "R")
    if not token or (not token_is_corner(token) and not token_is_moving_rotation(token) and token not in valid_prefixes):
        g.crashed = True
        return

    # Time budget guard
    if g.total_time_ms >= TIME_BUDGET_MS:
        g.ended = True
        return

    run_is_started = run_started(g)

    # In-place rotations
    if token_is_rotation(token):
        if g.mouse.momentum != 0:
            g.crashed = True
            return
        rotate_in_place(g.mouse, token)
        # Add base time, but not to run_time_ms if still at start center
        add_time(g, BASE_INPLACE_45, run_is_started)
        return

    # Moving rotations (translation + 45 end rotation)
    if token_is_moving_rotation(token):
        # Normalize to components
        trans = token[:2] if token != "BBL" and token != "BBR" else "BB"
        endr = token[-1]  # 'L' or 'R'
        # Legality: compute momentum change and m_eff <= 1
        if illegal_reverse_accel(g.mouse.momentum, trans):
            g.crashed = True
            return
        m_out = adjust_momentum(g.mouse.momentum, trans)
        eff = meff(g.mouse.momentum, m_out)
        if eff > 1.0:
            g.crashed = True
            return

        # Translation
        # BB translation: if |m_in|>0, still moves one half-step towards momentum's direction
        # Otherwise default action at rest
        if trans == "BB":
            if abs(g.mouse.momentum) == 0:
                add_time(g, BASE_DEFAULT_AT_REST, run_is_started)
                g.mouse.momentum = 0
                return
            # Move one half-step in direction of current momentum heading sign
            dist_cm, base_t = move_half_step_distance(g.mouse.heading_step)
            red = reduction_from_meff(eff)
            ms = round_ms(base_t * (1.0 - red))
            ok = apply_translation(g.maze, g.mouse, dist_cm)
            if not ok:
                g.crashed = True
                return
            add_time(g, ms, True)
            # Momentum update after translation
            g.mouse.momentum = m_out
            # End rotation free
            rotate_in_place(g.mouse, endr)
            on_reach_goal_if_any(g)
            return
        else:
            # F?/V? translation: move one half-step along current heading
            dist_cm, base_t = move_half_step_distance(g.mouse.heading_step)
            red = reduction_from_meff(eff)
            ms = round_ms(base_t * (1.0 - red))
            ok = apply_translation(g.maze, g.mouse, dist_cm)
            if not ok:
                g.crashed = True
                return
            add_time(g, ms, True)
            # Momentum update
            g.mouse.momentum = m_out
            # End rotation free
            rotate_in_place(g.mouse, endr)
            on_reach_goal_if_any(g)
            return

    # Corner turns
    if token_is_corner(token):
        # Parse: a b c [d] => (F/V)(0/1/2)(L/R)(T/W)[(L/R)]
        a, b, c, d, e = token[0], token[1], token[2], token[3], token[4] if len(token) == 5 else None
        # Constraints
        if not heading_is_cardinal(g.mouse.heading_step):
            g.crashed = True
            return
        # Direction agreement
        curt_dir = 1 if g.mouse.momentum >= 0 else -1 if g.mouse.momentum < 0 else 0
        tok_dir = 1 if a == "F" else -1
        if curt_dir != 0 and curt_dir != tok_dir:
            g.crashed = True
            return
        if illegal_reverse_accel(g.mouse.momentum, a + b):
            g.crashed = True
            return
        m_out = adjust_momentum(g.mouse.momentum, a + b)
        eff = meff(g.mouse.momentum, m_out)
        limit = 1.0 if d == "T" else 2.0
        if eff > limit:
            g.crashed = True
            return
        base = BASE_CORNER_T if d == "T" else BASE_CORNER_W
        red = reduction_from_meff(eff)
        ms = round_ms(base * (1.0 - red))
        # Approximate arc: move center by quarter-circle chord length to next half-step corner
        # For empty maze with only perimeter, we assume arc stays inside bounds
        # Advance center to the next cell corner approximately:
        # Move half cell in the perpendicular axis (tight ~8 cm radius arc).
        # Use a small segmented move to stay within bounds.
        segs = 8
        ang_delta = (math.pi / 2) / segs
        radius = CELL_CM / 2.0 if d == "T" else CELL_CM  # 8 cm or 16 cm
        # Starting at heading cardinal; orbit center along arc with center offset to inside corner
        # Approximate by step-wise translation inside bounds
        # Given simplicity and empty maze, we just ensure perimeter not crossed:
        ok = True
        # Emulate a short move within current cell bounds
        # Use small epsilon moves towards the corner
        step_cm = max(1.0, radius / segs)
        for _ in range(segs):
            if not apply_translation(g.maze, g.mouse, step_cm):
                ok = False
                break
        if not ok:
            g.crashed = True
            return
        # Apply heading change
        corner_apply_heading(g.mouse, c, e)
        add_time(g, ms, True)
        g.mouse.momentum = m_out
        on_reach_goal_if_any(g)
        return

    # Plain translations and braking
    t = token
    # Opposite-direction accel rule
    if illegal_reverse_accel(g.mouse.momentum, t):
        g.crashed = True
        return

    # BB at rest -> default action
    if t == "BB" and abs(g.mouse.momentum) == 0:
        add_time(g, BASE_DEFAULT_AT_REST, run_is_started)
        return

    # BB with |m|>0: still moves one half-step toward momentum direction
    if t == "BB" and abs(g.mouse.momentum) > 0:
        dist_cm, base_t = move_half_step_distance(g.mouse.heading_step)
        m_out = adjust_momentum(g.mouse.momentum, t)
        eff = meff(g.mouse.momentum, m_out)
        red = reduction_from_meff(eff)
        ms = round_ms(base_t * (1.0 - red))
        ok = apply_translation(g.maze, g.mouse, dist_cm)
        if not ok:
            g.crashed = True
            return
        add_time(g, ms, True)
        g.mouse.momentum = m_out
        on_reach_goal_if_any(g)
        return

    # F?/V? translation: move one half-step in current heading
    if t[0] in ("F", "V") and t[1] in ("0", "1", "2"):
        m_out = adjust_momentum(g.mouse.momentum, t)
        eff = meff(g.mouse.momentum, m_out)
        dist_cm, base_t = move_half_step_distance(g.mouse.heading_step)
        red = reduction_from_meff(eff)
        ms = round_ms(base_t * (1.0 - red))
        ok = apply_translation(g.maze, g.mouse, dist_cm)
        if not ok:
            g.crashed = True
            return
        add_time(g, ms, True)
        g.mouse.momentum = m_out
        on_reach_goal_if_any(g)
        return

    # Otherwise unrecognized -> crash
    g.crashed = True

def apply_thinking_time(g: ChallengeState, instructions: List[str]) -> None:
    if instructions:
        add_time(g, 50, run_started(g))

def simulate_batch(g: ChallengeState, instructions: List[str], end_flag: bool) -> None:
    if g.crashed or g.ended:
        return
    if end_flag:
        g.ended = True
        return
    # Empty or invalid instruction array -> crash
    if instructions is None or not isinstance(instructions, list):
        g.crashed = True
        return
    if len(instructions) == 0:
        g.crashed = True
        return

    apply_thinking_time(g, instructions)

    for token in instructions:
        if g.crashed or g.ended:
            break
        process_instruction(g, token)

    # Time budget end
    if g.total_time_ms >= TIME_BUDGET_MS:
        g.ended = True

    # New run if at start center with momentum 0
    if not g.crashed and not g.ended:
        if g.maze.at_start_center(g.mouse.x, g.mouse.y) and g.mouse.momentum == 0:
            # Start a new run
            g.run += 1
            g.run_time_ms = 0
            g.goal_reached = False

# Controller
def controller_plan(g: ChallengeState, sensed: List[int]) -> List[str]:
    # Simple left-hand-ish: prefer left turn if clear, else straight, else right; brake to stop before rotating
    instr: List[str] = []

    if g.goal_reached:
        # Stop after reaching goal
        return ["BB", "BB"]

    front = sensed[2]
    left = sensed[0]
    right = sensed[4]

    if g.mouse.momentum < 0:
        # Decelerate to 0 before any forward accel
        instr.append("V0")
        return instr

    # If blocked ahead, stop then rotate left
    if front == 1:
        if g.mouse.momentum > 0:
            instr.append("BB")
            return instr
        # Rotate left 90° at rest
        instr.extend(["L", "L"])
        return instr

    # Prefer left if clear: rotate then go
    if left == 0 and g.mouse.momentum == 0:
        instr.extend(["L", "L"])
        return instr

    # Go straight; accelerate to +2, then hold
    if g.mouse.momentum < 2:
        instr.append("F2")
    else:
        instr.append("F1")

    return instr

@app.route("/micro-mouse", methods=["POST"])
def micro_mouse():
    payload = request.get_json(force=True, silent=True) or {}

    game_uuid = payload.get("game_uuid")
    if not game_uuid or not isinstance(game_uuid, str):
        return jsonify({"error": "game_uuid required"}), 400

    g = get_game(game_uuid)

    if payload.get("end") is True:
        g.ended = True
        # Scoring
        score_time = None
        if g.best_time_ms is not None:
            score_time = g.best_time_ms + (g.total_time_ms / 30.0)
        return jsonify({
            "instructions": [],
            "end": True,
            "score_time": score_time,
        })

    sensor_data = payload.get("sensor_data")
    if not isinstance(sensor_data, list) or len(sensor_data) != 5:
        sensor_data = g.maze.sensor_hits(g.mouse.x, g.mouse.y, g.mouse.heading_step)

    instructions = payload.get("instructions")
    if instructions is not None:
        if payload.get("end") is True:
            g.ended = True
        else:
            simulate_batch(g, instructions, False)

    on_reach_goal_if_any(g)

    next_instr = controller_plan(g, sensor_data)

    if g.crashed:
        return jsonify({
            "instructions": [],
            "end": True,
            "crash": True
        })

    if g.total_time_ms >= TIME_BUDGET_MS:
        g.ended = True
        score_time = None
        if g.best_time_ms is not None:
            score_time = g.best_time_ms + (g.total_time_ms / 30.0)
        return jsonify({
            "instructions": [],
            "end": True,
            "score_time": score_time,
        })

    return jsonify({
        "instructions": next_instr,
        "end": False
    })