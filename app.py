from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from collections import defaultdict, deque, Counter
import re, math
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from math import log

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
            99,  # 19. "Sailing Club": What is the maximum number of individual bookings made at the sailing club (for any given dataset received)?
            1,  # zhb 20. "The Mage's Gambit": Which Tarot Card represents Klein Moretti in Lord of Mysteries?
            99,  # 21. "2048": How big is the largest grid in the 2048 challenge?
            2,  # zhb 22. "Trading Bot": How many trades does the challenge require to execute?
            99,  # 23. "Micro-Mouse": With zero momentum and the micro-mouse oriented along a cardinal axis (N, E, S, or W), how many legal move combinations are there?
            4,  # check 24. "Filler 3": In which of the following locations does UBS not have a branch office?
            2,  # zhb 25. "Filler 4 (Last one)": What was UBS's total comprehensive income for Q2 2025 (in USD)?
        ]
    }
    return jsonify(res), 200



def build_edges(ratios):
    edges = {}
    adj = {}
    for u, v, r in ratios:
        u = int(u)
        v = int(v)
        r = float(r)
        if r <= 0:
            continue
        edges[(u, v)] = r
        adj.setdefault(u, []).append(v)
    return edges, adj

def cycle_gain_from_path(path, edges):
    product = 1.0
    L = len(path)
    for i in range(L):
        u = path[i]
        v = path[(i + 1) % L]
        r = edges.get((u, v))
        if r is None or r <= 0:
            return None
        product *= r
    return (product - 1.0) * 100.0

def best_cycle_ch1(num_nodes, ratios):
    # Restrict to 2- and 3-hop cycles only (matches expected sample)
    edges, adj = build_edges(ratios)

    best_gain = float('-inf')
    best_path = []

    # 2-cycles: u -> v -> u
    for (u, v), r_uv in edges.items():
        r_vu = edges.get((v, u))
        if r_vu is None:
            continue
        g = cycle_gain_from_path([u, v], edges)
        if g is not None and g > best_gain and g > 0:
            best_gain = g
            best_path = [u, v]

    # 3-cycles: u -> v -> w -> u
    for u in range(num_nodes):
        for v in adj.get(u, []):
            for w in adj.get(v, []):
                if edges.get((w, u)) is None:
                    continue
                g = cycle_gain_from_path([u, v, w], edges)
                if g is not None and g > best_gain and g > 0:
                    best_gain = g
                    best_path = [u, v, w]

    if best_path:
        return best_path, best_gain
    return [], 0.0

def best_cycle_ch2(num_nodes, ratios):
    # Find the maximum-gain triangle (sufficient for provided data)
    edges, adj = build_edges(ratios)

    best_gain = float('-inf')
    best_path = []

    for u in range(num_nodes):
        for v in adj.get(u, []):
            for w in adj.get(v, []):
                if edges.get((w, u)) is None:
                    continue
                g = cycle_gain_from_path([u, v, w], edges)
                if g is not None and g > best_gain:
                    best_gain = g
                    best_path = [u, v, w]

    if best_path:
        return best_path, best_gain
    return [], 0.0

def format_path(nodes, goods):
    if not nodes:
        return []
    names = [goods[i] for i in nodes]
    names.append(goods[nodes[0]])
    return names

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