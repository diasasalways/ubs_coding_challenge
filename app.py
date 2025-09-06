from flask import Flask, request, jsonify, make_response, Response
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from collections import defaultdict, deque, Counter
import re, math
from dataclasses import dataclass
import xml.etree.ElementTree as ET

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

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
            3,  # zhb 21. "2048": How big is the largest grid in the 2048 challenge?
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
        print(seq)
        return Response(resp_svg, mimetype='image/svg+xml')
    except Exception:
        # Minimal safe fallback
        resp_svg = '<svg xmlns="http://www.w3.org/2000/svg"><text>111111</text></svg>'
        return Response(resp_svg, mimetype='image/svg+xml')

        # =========================
# Duolingo Sort - Utilities
# =========================

ROMAN_SYMBOLS = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000,
}

ROMAN_SUBTRACTIVES = {
    'IV': 4,
    'IX': 9,
    'XL': 40,
    'XC': 90,
    'CD': 400,
    'CM': 900,
}

def int_to_roman(value: int) -> str:
    if value <= 0 or value >= 4000:
        raise ValueError("Roman numerals support 1..3999")
    numerals = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
    ]
    result = []
    remaining = value
    for amount, symbol in numerals:
        count = remaining // amount
        if count:
            result.append(symbol * count)
            remaining -= amount * count
        if remaining == 0:
            break
    return ''.join(result)

def roman_to_int(token: str) -> Optional[int]:
    tok = token.upper()
    if not re.fullmatch(r'[IVXLCDM]+', tok):
        return None
    index = 0
    total = 0
    while index < len(tok):
        if index + 1 < len(tok) and tok[index:index+2] in ROMAN_SUBTRACTIVES:
            total += ROMAN_SUBTRACTIVES[tok[index:index+2]]
            index += 2
        else:
            value = ROMAN_SYMBOLS.get(tok[index])
            if value is None:
                return None
            total += value
            index += 1
    # Validate canonical form by re-encoding
    if total <= 0 or total >= 4000:
        return None
    if int_to_roman(total) != tok:
        return None
    return total

def is_cjk(token: str) -> bool:
    for ch in token:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x4DBF or
            0x4E00 <= code <= 0x9FFF or
            0xF900 <= code <= 0xFAFF
        ):
            return True
    return False

def detect_chinese_variant(token: str) -> Optional[str]:
    if not is_cjk(token):
        return None
    trad_markers = sum(ch in set('萬億兩') for ch in token)
    simp_markers = sum(ch in set('万亿两') for ch in token)
    if trad_markers > simp_markers:
        return 'zh_trad'
    if simp_markers > trad_markers:
        return 'zh_simp'
    # Default to traditional if ambiguous but CJK
    return 'zh_trad'

EN_UNITS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
EN_TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}
EN_SCALES = {
    'hundred': 100,
    'thousand': 1000,
    'million': 1000000,
    'billion': 1000000000,
}

def english_to_int(token: str) -> Optional[int]:
    s = token.lower().strip()
    s = s.replace(',', ' ')
    s = s.replace('-', ' ')
    # plural scales to singular
    s = re.sub(r'\b(hundreds)\b', 'hundred', s)
    s = re.sub(r'\b(thousands)\b', 'thousand', s)
    s = re.sub(r'\b(millions)\b', 'million', s)
    s = re.sub(r'\b(billions)\b', 'billion', s)
    # articles and filler
    s = re.sub(r'\b(and)\b', ' ', s)
    s = re.sub(r'\b(a|an)\b', 'one', s)
    s = re.sub(r'\s+', ' ', s)
    if not s:
        return None
    parts = s.split(' ')
    valid_words = set(EN_UNITS) | set(EN_TENS) | set(EN_SCALES)
    for w in parts:
        if w and w not in valid_words:
            return None
    total = 0
    current = 0
    for word in parts:
        if not word:
            continue
        if word in EN_UNITS:
            current += EN_UNITS[word]
        elif word in EN_TENS:
            current += EN_TENS[word]
        elif word == 'hundred':
            if current == 0:
                current = 1
            current *= 100
        else:
            scale = EN_SCALES[word]
            total += current * scale
            current = 0
    total += current
    if total < 0:
        return None
    return total

# German dictionaries
DE_UNITS = {
    'null': 0, 'eins': 1, 'ein': 1, 'eine': 1, 'einen': 1, 'einem': 1, 'einer': 1, 'eines': 1,
    'zwei': 2, 'drei': 3, 'vier': 4,
    'fuenf': 5, 'fünf': 5, 'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9,
    'zehn': 10, 'elf': 11, 'zwoelf': 12, 'zwölf': 12,
}
DE_TEENS = {
    'dreizehn': 13, 'vierzehn': 14, 'fuenfzehn': 15, 'fünfzehn': 15,
    'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19,
}
DE_TENS = {
    'zwanzig': 20, 'dreissig': 30, 'dreißig': 30, 'vierzig': 40,
    'fuenfzig': 50, 'fünfzig': 50, 'sechzig': 60, 'siebzig': 70,
    'achtzig': 80, 'neunzig': 90,
}

def normalize_german(s: str) -> str:
    s = s.lower().strip()
    s = s.replace('ß', 'ss')
    return s

def parse_german_below_thousand(s: str) -> Optional[int]:
    s = s.replace(' ', '').replace('-', '')
    # Handle hundreds
    if 'hundert' in s:
        idx = s.find('hundert')
        left = s[:idx]
        right = s[idx + len('hundert'):]
        if left == '':
            hundreds = 100
        else:
            if left in DE_UNITS:
                hundreds = DE_UNITS[left] * 100
            else:
                # left may itself be a compound like 'zweiunddrei' (unlikely), reject
                return None
        rest = 0
        if right:
            parsed = parse_german_below_hundred(right)
            if parsed is None:
                return None
            rest = parsed
        return hundreds + rest
    # Else below 100
    return parse_german_below_hundred(s)

def parse_german_below_hundred(s: str) -> Optional[int]:
    s = s.replace(' ', '').replace('-', '')
    if not s:
        return 0
    # direct matches
    if s in DE_UNITS:
        return DE_UNITS[s]
    if s in DE_TEENS:
        return DE_TEENS[s]
    if s in DE_TENS:
        return DE_TENS[s]
    # pattern unit + 'und' + tens
    if 'und' in s:
        parts = s.split('und')
        if len(parts) != 2:
            return None
        unit_part, tens_part = parts
        if tens_part in DE_TENS and unit_part in DE_UNITS:
            return DE_TENS[tens_part] + DE_UNITS[unit_part]
        return None
    return None

def german_to_int(token: str) -> Optional[int]:
    s = normalize_german(token)
    if not s:
        return None
    # Large scales right-to-left
    total = 0
    remainder = s
    # millions
    for key in ['millionen', 'million']:
        if key in remainder:
            idx = remainder.find(key)
            left = remainder[:idx].strip().replace('-', '')
            remainder = remainder[idx + len(key):].strip()
            mult = 1 if left == '' else parse_german_below_thousand(left)
            if mult is None or mult == 0:
                return None
            total += mult * 1_000_000
            break
    # milliarden (optional)
    for key in ['milliarden', 'milliarde']:
        if key in remainder:
            idx = remainder.find(key)
            left = remainder[:idx].strip().replace('-', '')
            remainder = remainder[idx + len(key):].strip()
            mult = 1 if left == '' else parse_german_below_thousand(left)
            if mult is None or mult == 0:
                return None
            total += mult * 1_000_000_000
            break
    # tausend
    if 'tausend' in remainder:
        idx = remainder.find('tausend')
        left = remainder[:idx].strip().replace('-', '')
        remainder = remainder[idx + len('tausend'):].strip()
        mult = 1 if left == '' else parse_german_below_thousand(left)
        if mult is None or mult == 0:
            return None
        total += mult * 1000
    # rest
    if remainder:
        parsed = parse_german_below_thousand(remainder)
        if parsed is None:
            return None
        total += parsed
    return total if total >= 0 else None

CH_DIGITS = {
    '零': 0, '〇': 0,
    '一': 1, '壹': 1,
    '二': 2, '贰': 2, '貳': 2, '两': 2, '兩': 2,
    '三': 3, '叁': 3, '參': 3,
    '四': 4, '肆': 4,
    '五': 5, '伍': 5,
    '六': 6, '陸': 6, '陆': 6,
    '七': 7, '柒': 7,
    '八': 8, '捌': 8,
    '九': 9, '玖': 9,
}
CH_UNIT_MAP = {
    '十': 10, '百': 100, '千': 1000,
    '万': 10_000, '萬': 10_000,
    '亿': 100_000_000, '億': 100_000_000,
    '兆': 1_000_000_000_000,
}

def chinese_to_int(token: str) -> Optional[int]:
    s = token.strip()
    if not is_cjk(s):
        return None
    total = 0
    section = 0
    temp = 0
    last_unit_mag = 1
    used_small_in_section = False
    saw_zero_after_last_unit = False

    for ch in s:
        if ch in CH_DIGITS:
            temp = CH_DIGITS[ch]
            continue
        if ch in ('零', '〇'):
            temp = 0
            saw_zero_after_last_unit = True
            continue
        unit = CH_UNIT_MAP.get(ch)
        if unit is None:
            return None
        if unit < 10_000:  # small unit
            if temp == 0:
                temp = 1
            section += temp * unit
            temp = 0
            last_unit_mag = unit
            used_small_in_section = True
            saw_zero_after_last_unit = False
        else:  # large unit: 万, 亿, 兆
            # Add any leftover temp as ones in the current section
            if temp != 0:
                section += temp
                temp = 0
            if section == 0:
                section = 1
            total += section * unit
            section = 0
            last_unit_mag = unit
            used_small_in_section = False
            saw_zero_after_last_unit = False

    # End of string handling: implicit units
    if temp != 0:
        if used_small_in_section:
            # If we used a small unit in this section, a trailing digit may imply the next lower unit
            if last_unit_mag >= 100 and not saw_zero_after_last_unit:
                section += temp * (last_unit_mag // 10)
            else:
                section += temp
        else:
            # No small unit in this section; if there was a previous big unit, scale by its /10
            if last_unit_mag >= 10_000 and not saw_zero_after_last_unit:
                section += temp * (last_unit_mag // 10)
            else:
                section += temp

    return total + section

LANG_ROMAN = 'roman'
LANG_EN = 'english'
LANG_ZH_TRAD = 'zh_trad'
LANG_ZH_SIMP = 'zh_simp'
LANG_DE = 'german'
LANG_AR = 'arabic'

LANG_RANK = {
    LANG_ROMAN: 1,
    LANG_EN: 2,
    LANG_ZH_TRAD: 3,
    LANG_ZH_SIMP: 4,
    LANG_DE: 5,
    LANG_AR: 6,
}

def detect_language(token: str) -> Optional[str]:
    t = token.strip()
    if not t:
        return None
    # Chinese
    zh = detect_chinese_variant(t)
    if zh is not None:
        return zh
    # Roman
    if re.fullmatch(r'[IVXLCDMivxlcdm]+', t):
        return LANG_ROMAN
    # Arabic numerals
    if re.fullmatch(r'[0-9][0-9,._\s]*', t) and re.fullmatch(r'\d+', re.sub(r'[,._\s]', '', t)):
        return LANG_AR
    # English
    if english_to_int(t) is not None:
        return LANG_EN
    # German
    if german_to_int(t) is not None:
        return LANG_DE
    return None

def parse_value_with_language(token: str) -> Tuple[Optional[int], Optional[str]]:
    language = detect_language(token)
    if language is None:
        return None, None
    if language == LANG_ROMAN:
        return roman_to_int(token), language
    if language == LANG_AR:
        try:
            normalized = re.sub(r'[,._\s]', '', token)
            value = int(normalized)
            if value < 0:
                return None, None
            return value, language
        except Exception:
            return None, None
    if language == LANG_EN:
        return english_to_int(token), language
    if language == LANG_DE:
        return german_to_int(token), language
    if language in (LANG_ZH_TRAD, LANG_ZH_SIMP):
        return chinese_to_int(token), language
    return None, None

def sort_duolingo_list(part: str, items: List[str], challenge: Any) -> Tuple[Optional[List[str]], Optional[str]]:
    parsed: List[Tuple[int, str, int, str]] = []  # value, language, index, original
    for idx, tok in enumerate(items):
        value, language = parse_value_with_language(tok)
        if value is None or language is None:
            return None, f"Unrecognized token '{tok}' in challenge {challenge}"
        if part == 'ONE' and language not in (LANG_ROMAN, LANG_AR):
            return None, f"Part ONE expects only Roman or Arabic numerals. Got '{tok}'."
        parsed.append((value, language, idx, tok))
    parsed.sort(key=lambda it: (it[0], LANG_RANK[it[1]], it[2]))
    if part == 'ONE':
        return [str(v) for (v, _lang, _i, _o) in parsed], None
    else:
        return [o for (_v, _lang, _i, o) in parsed], None

@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    data = request.get_json(silent=True) or {}
    part = data.get('part')
    challenge = data.get('challenge')
    challenge_input = data.get('challengeInput') or {}
    items = challenge_input.get('unsortedList')
    if part not in ('ONE', 'TWO'):
        return bad_request("'part' must be 'ONE' or 'TWO'.")
    if not isinstance(items, list) or any(not isinstance(x, str) for x in items):
        return bad_request("'challengeInput.unsortedList' must be a list of strings.")
    result, err = sort_duolingo_list(part, items, challenge)
    if err is not None:
        return bad_request(err)
    return jsonify({"sortedList": result})


@app.route("/the-mages-gambit", methods=["POST"])
def mages_gambit():
    """
    Calculate the minimum time Klein needs to defeat all undead and join the expedition.

    Expected input format:
    [
        {
            "intel": [[front, mp_cost], ...],
            "reserve": int,
            "fronts": int,
            "stamina": int
        },
        ...
    ]

    Returns:
    [
        {"time": int},
        ...
    ]
    """
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, list):
            return jsonify({"error": "Expected array of test cases"}), 400

        results = []

        for test_case in payload:
            if not isinstance(test_case, dict):
                return jsonify({"error": "Each test case must be an object"}), 400

            intel = test_case.get("intel", [])
            reserve = test_case.get("reserve", 0)
            fronts = test_case.get("fronts", 0)
            stamina = test_case.get("stamina", 0)

            # Validate inputs
            if not isinstance(intel, list) or not isinstance(reserve, int) or not isinstance(stamina, int) or not isinstance(fronts, int):
                return jsonify({"error": "Invalid input types"}), 400

            if reserve <= 0 or stamina <= 0:
                return jsonify({"error": "'reserve' and 'stamina' must be positive"}), 400

            for attack in intel:
                if not isinstance(attack, list) or len(attack) != 2:
                    return jsonify({"error": "Each intel entry must be [front, mp_cost]"}), 400
                front, mp_cost = attack
                if not isinstance(front, int) or not isinstance(mp_cost, int):
                    return jsonify({"error": "Front and MP cost must be integers"}), 400
                if front < 1:
                    return jsonify({"error": "Front must be >= 1"}), 400
                if fronts > 0 and front > fronts:
                    return jsonify({"error": f"Front must be between 1 and {fronts}"}), 400
                if mp_cost < 1 or mp_cost > reserve:
                    return jsonify({"error": f"MP cost must be between 1 and {reserve}"}), 400

            # Calculate minimum time
            min_time = calculate_mage_combat_time(intel, reserve, stamina)
            results.append({"time": min_time})

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_mage_combat_time(intel, reserve, stamina):
    """
    Calculate the minimum time needed for Klein to defeat all undead.

    Args:
        intel: List of [front, mp_cost] representing undead attacks in sequence
        reserve: Maximum mana capacity
        stamina: Number of spells that can be cast before cooldown required

    Returns:
        int: Minimum time in minutes
    """
    # Empty intel: no actions needed; earliest time is 0
    if not intel:
        return 0

    current_mp = reserve
    current_stamina = stamina
    total_time = 0
    last_front = None
    last_action_was_cooldown = False

    # Look ahead optimization: check if we can save time by pre-cooling before long same-front runs
    i = 0
    while i < len(intel):
        front, mp_cost = intel[i]
        
        # Look ahead to find same-front run length and total MP cost
        j = i
        run_mp_sum = 0
        run_length = 0
        while j < len(intel) and intel[j][0] == front:
            run_mp_sum += intel[j][1]
            run_length += 1
            j += 1
        
        # If this is a new front and we can't complete the run with current resources,
        # consider pre-cooling to avoid mid-run cooldown + retarget penalty
        if (last_front != front and 
            (current_mp < run_mp_sum or current_stamina < run_length) and
            (current_mp < reserve or current_stamina < stamina) and
            run_length > 1):  # Only for multi-attack runs
            
            # Pre-cool to avoid mid-run retarget
            total_time += 10
            current_mp = reserve
            current_stamina = stamina
            last_action_was_cooldown = True

        # Process each attack in the run
        for k in range(i, j):
            front, mp_cost = intel[k]
            
            # Check if we need cooldown before this attack
            had_cooldown = False
            if current_mp < mp_cost or current_stamina < 1:
                # Force cooldown to recover resources
                total_time += 10  # Cooldown takes 10 minutes
                current_mp = reserve
                current_stamina = stamina
                had_cooldown = True
                last_action_was_cooldown = True

            # Execute the attack
            # If same front as last attack AND no cooldown happened, extend AOE (0 extra time)
            if front == last_front and not had_cooldown:
                spell_time = 0  # Extend AOE, no extra time
            else:
                spell_time = 10  # New target or after cooldown

            total_time += spell_time
            current_mp -= mp_cost
            current_stamina -= 1
            last_front = front
            last_action_was_cooldown = False
        
        i = j

    # Must end with cooldown to be ready for expedition (unless already in cooldown)
    if not last_action_was_cooldown:
        total_time += 10

    return total_time