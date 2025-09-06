from flask import Flask, request, jsonify, make_response, Response
from typing import Any, Dict, List, Tuple, Optional, Set
from scipy.stats import linregress
from scipy import interpolate
import numpy as np
from collections import defaultdict, deque, Counter
import re, math, threading, time
import uuid
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.json.sort_keys = False

def bad_request(message: str):
    resp = make_response(jsonify({"error": message}), 400)
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route('/')
def root():
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

def is_vowel(char):
    return char.lower() in 'aeiou'

def is_consonant(char):
    return char.isalpha() and not is_vowel(char)

def only_letters(s):
    return ''.join(ch for ch in s if ch.isalpha())

def to_alpha_upper(s):
    return ''.join(ch for ch in s.upper() if 'A' <= ch <= 'Z')

# =========================
# Challenge 1: Reverse Obfuscation Analysis
# =========================

def atbash_char(c):
    if 'A' <= c <= 'Z':
        return chr(ord('Z') - (ord(c) - ord('A')))
    if 'a' <= c <= 'z':
        return chr(ord('z') - (ord(c) - ord('a')))
    return c

def atbash_string(s):
    return ''.join(atbash_char(c) for c in s)

def reverse_each_word_preserve_ws(s):
    parts = re.split(r'(\s+)', s)
    out = []
    for p in parts:
        if p.isspace() or p == '':
            out.append(p)
        else:
            out.append(p[::-1])
    return ''.join(out)

def process_per_word(s, func):
    parts = re.split(r'(\s+)', s)
    out = []
    for p in parts:
        if p.isspace() or p == '':
            out.append(p)
        else:
            out.append(func(p))
    return ''.join(out)

def swap_pairs_in_word(word):
    chars = list(word)
    n = len(chars)
    i = 0
    while i + 1 < n:
        chars[i], chars[i+1] = chars[i+1], chars[i]
        i += 2
    return ''.join(chars)

def inverse_encode_index_parity_word(word):
    n = len(word)
    if n == 0:
        return word
    e = (n + 1) // 2
    even = word[:e]
    odd = word[e:]
    res = []
    ie = 0
    io = 0
    for i in range(n):
        if i % 2 == 0:
            res.append(even[ie]); ie += 1
        else:
            res.append(odd[io]); io += 1
    return ''.join(res)

def collapse_double_consonants_word(word):
    res = []
    i = 0
    n = len(word)
    while i < n:
        c = word[i]
        if i + 1 < n and is_consonant(c) and word[i+1] == c:
            res.append(c)
            i += 2
        else:
            res.append(c)
            i += 1
    return ''.join(res)

def reverse_transformations(transformations, transformed_input):
    # Normalize to list of strings per clarification
    if isinstance(transformed_input, str):
        items = [transformed_input]
    elif isinstance(transformed_input, list):
        items = transformed_input[:]
    else:
        return ""

    # Inverses
    def inv_mirror_words(x):
        return reverse_each_word_preserve_ws(x)

    def inv_encode_mirror_alphabet(x):
        return atbash_string(x)

    def inv_toggle_case(x):
        return x.swapcase()

    def inv_swap_pairs(x):
        return process_per_word(x, swap_pairs_in_word)

    def inv_encode_index_parity(x):
        return process_per_word(x, inverse_encode_index_parity_word)

    def inv_double_consonants(x):
        return process_per_word(x, collapse_double_consonants_word)

    inverses = {
        "mirror_words": inv_mirror_words,
        "encode_mirror_alphabet": inv_encode_mirror_alphabet,
        "toggle_case": inv_toggle_case,
        "swap_pairs": inv_swap_pairs,
        "encode_index_parity": inv_encode_index_parity,
        "double_consonants": inv_double_consonants
    }

    # Parse function names like "encode_mirror_alphabet(x)"
    def parse_name(t):
        t = t.strip()
        m = re.match(r'^([a-zA-Z_]+)\s*\(', t)
        return m.group(1) if m else t

    ops = [parse_name(t) for t in (transformations or []) if isinstance(t, str)]
    # Apply inverses in reverse order
    for op in reversed(ops):
        func = inverses.get(op)
        if not func:
            continue
        items = [func(s) for s in items]

    return ' '.join(items).strip()

# =========================
# Challenge 2: Coordinate digit recognition
# =========================

def extract_number_from_coordinates(coords):
    """Extract digit from coordinate patterns"""
    if not coords:
        return "0"
    
    # Convert to float pairs
    points = []
    for c in coords:
        if len(c) >= 2:
            try:
                lat = float(c[0])
                lon = float(c[1])
                points.append((lat, lon))
            except:
                continue
    
    if len(points) < 3:
        return "0"
    
    # Remove obvious outliers using median distance
    if len(points) > 5:
        # Calculate centroid
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        
        # Calculate distances from centroid
        distances = [math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) for p in points]
        median_dist = sorted(distances)[len(distances)//2]
        
        # Keep points within 3x median distance
        filtered = []
        for i, d in enumerate(distances):
            if d <= 3 * median_dist:
                filtered.append(points[i])
        
        if len(filtered) >= 3:
            points = filtered
    
    # Normalize coordinates to grid
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    
    if max_x == min_x or max_y == min_y:
        return "1"
    
    # Map to 5x7 grid
    grid = [[0 for _ in range(5)] for _ in range(7)]
    
    for px, py in points:
        gx = int(4 * (px - min_x) / (max_x - min_x))
        gy = int(6 * (py - min_y) / (max_y - min_y))
        gx = max(0, min(4, gx))
        gy = max(0, min(6, gy))
        grid[gy][gx] = 1
    
    # Simple pattern matching for digits 0-9
    grid_str = ''.join(''.join(str(cell) for cell in row) for row in grid)
    
    # Simplified digit patterns
    patterns = {
        "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        "6": ["01110", "10001", "10000", "11110", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "10000", "10000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "10001", "01110"]
    }
    
    # Find best match
    best_digit = "0"
    best_score = -1
    
    for digit, pattern in patterns.items():
        score = 0
        for i in range(7):
            for j in range(5):
                if grid[i][j] == int(pattern[i][j]):
                    score += 1
        
        if score > best_score:
            best_score = score
            best_digit = digit
    
    return best_digit

# =========================
# Challenge 3: Operational Intelligence Extraction
# =========================

def parse_log_entry(log_text):
    """Parse log entry to extract cipher type and payload"""
    # Parse key-value pairs from log
    fields = {}
    
    # Split by | and parse each part
    parts = log_text.split('|')
    for part in parts:
        part = part.strip()
        if ':' in part:
            key, value = part.split(':', 1)
            fields[key.strip().upper()] = value.strip()
    
    # Extract cipher type
    cipher_type = None
    for key in ['CIPHER_TYPE', 'CIPHER', 'METHOD', 'TYPE']:
        if key in fields:
            cipher_type = fields[key].upper()
            break
    
    # Handle cipher aliases
    if cipher_type in ['ROTATION_CIPHER', 'ROT', 'CAESAR']:
        cipher_type = 'CAESAR'
    elif cipher_type in ['RAILFENCE', 'RAIL_FENCE']:
        cipher_type = 'RAILFENCE'
    elif cipher_type in ['KEYWORD', 'SUBSTITUTION']:
        cipher_type = 'KEYWORD'
    elif cipher_type in ['POLYBIUS', 'POLYBIUS_SQUARE']:
        cipher_type = 'POLYBIUS'
    
    # Extract payload
    payload = None
    for key in ['ENCRYPTED_PAYLOAD', 'PAYLOAD', 'DATA', 'MESSAGE', 'BODY']:
        if key in fields:
            payload = fields[key]
            break
    
    return cipher_type, payload

def decrypt_railfence(ciphertext, rails=3):
    """Decrypt rail fence cipher with 3 rails"""
    if rails <= 1:
        return ciphertext
    
    n = len(ciphertext)
    if n == 0:
        return ""
    
    # Calculate pattern and rail lengths
    cycle = 2 * rails - 2
    rail_lengths = [0] * rails
    
    for i in range(n):
        pos = i % cycle
        if pos < rails:
            rail_lengths[pos] += 1
        else:
            rail_lengths[cycle - pos] += 1
    
    # Split ciphertext into rails
    rails_text = []
    start = 0
    for length in rail_lengths:
        rails_text.append(ciphertext[start:start + length])
        start += length
    
    # Reconstruct plaintext
    result = []
    rail_indices = [0] * rails
    
    for i in range(n):
        pos = i % cycle
        if pos < rails:
            rail_idx = pos
        else:
            rail_idx = cycle - pos
        
        result.append(rails_text[rail_idx][rail_indices[rail_idx]])
        rail_indices[rail_idx] += 1
    
    return ''.join(result)

def decrypt_keyword(ciphertext, keyword="SHADOW"):
    """Decrypt keyword substitution cipher"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Build cipher alphabet
    keyword = keyword.upper()
    cipher_alpha = ""
    seen = set()
    
    # Add keyword letters first
    for ch in keyword:
        if ch.isalpha() and ch not in seen:
            cipher_alpha += ch
            seen.add(ch)
    
    # Add remaining letters
    for ch in alphabet:
        if ch not in seen:
            cipher_alpha += ch
    
    # Create decryption mapping
    decrypt_map = {}
    for i in range(26):
        decrypt_map[cipher_alpha[i]] = alphabet[i]
    
    # Decrypt
    result = []
    for ch in ciphertext.upper():
        if ch in decrypt_map:
            result.append(decrypt_map[ch])
        else:
            result.append(ch)
    
    return ''.join(result)

def decrypt_polybius(ciphertext):
    """Decrypt Polybius square cipher (5x5, I/J combined)"""
    square = [
        ['A', 'B', 'C', 'D', 'E'],
        ['F', 'G', 'H', 'I', 'K'],
        ['L', 'M', 'N', 'O', 'P'],
        ['Q', 'R', 'S', 'T', 'U'],
        ['V', 'W', 'X', 'Y', 'Z']
    ]
    
    # Extract digit pairs
    digits = re.findall(r'\d', ciphertext)
    if len(digits) % 2 != 0:
        return ""
    
    result = []
    for i in range(0, len(digits), 2):
        row = int(digits[i]) - 1
        col = int(digits[i+1]) - 1
        
        if 0 <= row < 5 and 0 <= col < 5:
            result.append(square[row][col])
    
    return ''.join(result)

def decrypt_caesar(ciphertext, shift=13):
    """Decrypt Caesar cipher (default ROT13)"""
    # Try to detect shift automatically for common words
    best_result = ""
    best_score = -1
    
    for s in range(26):
        result = ""
        for ch in ciphertext.upper():
            if 'A' <= ch <= 'Z':
                result += chr((ord(ch) - ord('A') - s) % 26 + ord('A'))
            else:
                result += ch
        
        # Score based on common English patterns
        score = 0
        words = result.split()
        for word in words:
            if word in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'WORD', 'WHAT', 'SAID', 'EACH', 'WHICH', 'SHE', 'DO', 'HOW', 'THEIR', 'IF', 'UP', 'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SO', 'SOME', 'HER', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'TIME', 'HAS', 'TWO', 'MORE', 'GO', 'NO', 'WAY', 'COULD', 'MY', 'THAN', 'FIRST', 'WATER', 'BEEN', 'CALL', 'WHO', 'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'DID', 'GET', 'COME', 'MADE', 'MAY', 'PART', 'FIREWALL', 'ATTACK', 'SECURITY', 'BANK', 'BREACH', 'DATA']:
                score += 10
        
        if score > best_score:
            best_score = score
            best_result = result
    
    return best_result if best_result else ciphertext

def decrypt_log_entry(log_text):
    """Parse and decrypt a log entry"""
    cipher_type, payload = parse_log_entry(log_text)
    
    if not cipher_type or not payload:
        return payload or ""
    
    if cipher_type == 'RAILFENCE':
        return decrypt_railfence(payload, 3)
    elif cipher_type == 'KEYWORD':
        return decrypt_keyword(payload, "SHADOW")
    elif cipher_type == 'POLYBIUS':
        return decrypt_polybius(payload)
    elif cipher_type == 'CAESAR':
        return decrypt_caesar(payload)
    else:
        return payload

# =========================
# Challenge 4: Final Communication Decryption
# =========================

def vigenere_decrypt(ciphertext, key):
    """Decrypt Vigenère cipher"""
    key = key.upper()
    result = []
    key_index = 0
    
    for ch in ciphertext.upper():
        if 'A' <= ch <= 'Z':
            shift = ord(key[key_index % len(key)]) - ord('A')
            decrypted_ch = chr((ord(ch) - ord('A') - shift) % 26 + ord('A'))
            result.append(decrypted_ch)
            key_index += 1
        else:
            result.append(ch)
    
    return ''.join(result)

def combine_keyword_and_number(keyword, number, ciphertext):
    """Combine keyword and number for final decryption"""
    if not keyword or not ciphertext:
        return ciphertext
    
    try:
        shift = int(number) if isinstance(number, str) else number
    except:
        shift = 0
    
    # Try Vigenère with keyword + Caesar shift
    step1 = vigenere_decrypt(ciphertext, keyword)
    
    # Apply Caesar shift
    result = []
    for ch in step1:
        if 'A' <= ch <= 'Z':
            result.append(chr((ord(ch) - ord('A') - shift) % 26 + ord('A')))
        else:
            result.append(ch)
    
    return ''.join(result)

def final_decryption(keyword, number, ciphertext):
    """Final decryption using recovered components"""
    if not ciphertext:
        return ""
    
    # Try different combinations
    attempts = [
        # Vigenère + Caesar
        combine_keyword_and_number(keyword, number, ciphertext),
        # Caesar + Vigenère  
        vigenere_decrypt(decrypt_caesar(ciphertext, int(str(number)) if number else 0), keyword),
        # Just Vigenère
        vigenere_decrypt(ciphertext, keyword) if keyword else ciphertext,
        # Just Caesar
        decrypt_caesar(ciphertext, int(str(number)) if number else 13),
    ]
    
    # Score each attempt
    best_result = ciphertext
    best_score = -1
    
    for attempt in attempts:
        score = 0
        words = attempt.split()
        for word in words:
            if len(word) >= 3 and word.upper() in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'WORD', 'WHAT', 'SAID', 'EACH', 'WHICH', 'SHE', 'HOW', 'THEIR', 'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SOME', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'TIME', 'HAS', 'TWO', 'MORE', 'WAY', 'COULD', 'THAN', 'FIRST', 'WATER', 'BEEN', 'CALL', 'WHO', 'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'DID', 'GET', 'COME', 'MADE', 'MAY', 'PART', 'OPERATION', 'SAFEGUARD', 'MERIDIAN', 'INTERNATIONAL', 'BANK', 'SECURITY', 'FIREWALL', 'ATTACK', 'BREACH', 'DATA', 'ENCRYPTION', 'PASSWORD', 'ACCESS', 'SYSTEM', 'ALERT', 'CRITICAL', 'TARGET', 'COMMAND', 'CONTROL', 'MESSAGE', 'GROUP', 'OBJECTIVE']:
                score += 10
        
        if score > best_score:
            best_score = score
            best_result = attempt
    
    return best_result

# =========================
# Flask endpoint
# =========================

@app.route('/operation-safeguard', methods=['POST'])
def operation_safeguard():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        # Challenge 1: Reverse transformations
        ch1_data = data.get('challenge_one', {})
        transformations = ch1_data.get('transformations', [])
        transformed_word = ch1_data.get('transformed_encrypted_word', '')
        
        challenge_one_result = reverse_transformations(transformations, transformed_word)

        # Challenge 2: Extract number from coordinates
        ch2_coords = data.get('challenge_two', [])
        challenge_two_result = extract_number_from_coordinates(ch2_coords)

        # Challenge 3: Decrypt log entry
        ch3_log = data.get('challenge_three', '')
        challenge_three_result = decrypt_log_entry(ch3_log)

        # Challenge 4: Final decryption using all components
        # Extract final ciphertext from the request
        final_ciphertext = data.get('challenge_four_ciphertext', '')
        if not final_ciphertext:
            # If not provided separately, try to extract from challenge results
            final_ciphertext = challenge_three_result
        
        challenge_four_result = final_decryption(
            challenge_one_result, 
            challenge_two_result, 
            final_ciphertext
        )

        return jsonify({
            'challenge_one': challenge_one_result,
            'challenge_two': challenge_two_result,
            'challenge_three': challenge_three_result,
            'challenge_four': challenge_four_result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500