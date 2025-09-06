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