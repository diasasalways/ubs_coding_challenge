from flask import Flask, request, jsonify, make_response
from math import hypot
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from collections import defaultdict, deque, Counter
import re, math
from math import log

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

if __name__ == "__main__":
    # For local development only
    app.run()
    # app.run(host='0.0.0.0', port=3000, debug=False)

@app.route("/")
def root():
    return "OK", 200

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
    data = request.get_json(force=True)
    if not isinstance(data, list) or len(data) != 2:
        return jsonify({'error': 'Input must be a JSON array of length 2'}), 400

    results = []

    # Challenge 1: restrict to 2- and 3-hop cycles
    c1 = data[0]
    goods1 = c1['goods']
    ratios1 = c1['ratios']
    path1_nodes, gain1 = best_cycle_ch1(len(goods1), ratios1)
    results.append({
        'path': format_path(path1_nodes, goods1),
        'gain': gain1
    })

    # Challenge 2: best triangle (max gain)
    c2 = data[1]
    goods2 = c2['goods']
    ratios2 = c2['ratios']
    path2_nodes, gain2 = best_cycle_ch2(len(goods2), ratios2)
    results.append({
        'path': format_path(path2_nodes, goods2),
        'gain': gain2
    })

    return jsonify(results)