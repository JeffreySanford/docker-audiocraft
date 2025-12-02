import os
import json
import difflib
import torch
import re
from typing import Dict, Tuple


def match_by_suffix(name, sd_keys, max_suffix=6):
    parts = name.split('.')
    max_len = min(max_suffix, len(parts))
    for suffix_len in range(max_len, 0, -1):
        suffix = '.'.join(parts[-suffix_len:])
        for key in sd_keys:
            if key.endswith(suffix):
                return key
    return None


def find_shape_match(name, param_tensor, sd):
    candidates = []
    pshape = tuple(param_tensor.shape)
    for k, v in sd.items():
        if tuple(v.shape) == pshape:
            candidates.append(k)
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if c.endswith(name.split('.')[-1]):
            return c
    best = None
    best_score = 0
    for c in candidates:
        rs = os.path.commonprefix(['.'.join(c.split('.')[::-1]), '.'.join(name.split('.')[::-1])])
        score = len(rs)
        if score > best_score:
            best_score = score
            best = c
    if best_score > 0:
        return best
    return None


def find_fuzzy_match(name, param_tensor, sd, top_n=5, ratio_thresh=0.75):
    keys = list(sd.keys())
    same_shape = [k for k, v in sd.items() if tuple(v.shape) == tuple(param_tensor.shape)]
    if same_shape:
        matches = difflib.get_close_matches(name, same_shape, n=top_n, cutoff=ratio_thresh)
        if matches:
            return matches[0]
    matches = difflib.get_close_matches(name, keys, n=top_n, cutoff=ratio_thresh)
    if matches:
        return matches[0]
    best = None
    best_score = 0.0
    from difflib import SequenceMatcher
    for k in keys:
        score = SequenceMatcher(None, name, k).ratio()
        if score > best_score:
            best_score = score
            best = k
    if best_score >= ratio_thresh:
        return best
    return None


def compute_match_score(name, candidate, param_tensor, sd):
    score = 0.0
    try:
        candidate_shape = tuple(sd[candidate].shape)
        param_shape = tuple(param_tensor.shape)
    except Exception:
        candidate_shape = None
        param_shape = tuple(param_tensor.shape)
    if candidate_shape is not None and candidate_shape == param_shape:
        score += 0.5
    if candidate.endswith(name.split('.')[-1]):
        score += 0.2

    def common_suffix_len(a, b):
        ra = a.split('.')[::-1]
        rb = b.split('.')[::-1]
        common = 0
        for x, y in zip(ra, rb):
            if x == y:
                common += 1
            else:
                break
        return common

    cs_len = common_suffix_len(name, candidate)
    score += min(0.1, 0.02 * cs_len)
    ratio = difflib.SequenceMatcher(None, name, candidate).ratio()
    score += 0.15 * ratio
    if score > 1.0:
        score = 1.0
    return score


def generate_transform_candidates(name):
    candidates = set()
    candidates.add(name)
    candidates.add(name.replace('.model.', '.'))
    candidates.add(name.replace('encoder.', 'encoder.model.'))
    candidates.add(name.replace('decoder.', 'decoder.model.'))
    candidates.add(name.replace('.conv.conv.', '.conv.'))
    candidates.add(name.replace('.conv.', '.conv.conv.'))
    candidates.add(name.replace('module.', ''))
    candidates.add(name.replace('.weight', '.weight_g'))
    candidates.add(name.replace('.weight', '.weight_v'))
    candidates.add(name.replace('.conv.conv.', '.conv.'))
    candidates.add(name.replace('.model.', '.'))
    if '.model.' in name:
        candidates.add(name.replace('.model.', '.'))
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p.isdigit():
            mutated = '.'.join(parts[:i] + parts[i+1:])
            candidates.add(mutated)
    return [c for c in sorted(candidates) if c]


def apply_mapping_to_module(module, sd, mapping, device='cpu'):
    applied = 0
    module_param_names = [n for n, _ in module.named_parameters()]
    sd_keys = set(sd.keys())
    for src, cand in mapping.items():
        if isinstance(cand, dict) and 'candidate' in cand:
            cand = cand['candidate']
        if src in module_param_names and cand in sd_keys:
            module_name, param_name = src.rsplit('.', 1) if '.' in src else ('', src)
            sub = module
            if module_name:
                for part in module_name.split('.'):
                    sub = getattr(sub, part)
            try:
                setattr(sub, param_name, torch.nn.Parameter(sd[cand].to(device)))
                applied += 1
            except Exception:
                pass
    return applied


def slugify(text: str, max_len: int = 120) -> str:
    """Create a safe filename-friendly slug from a title."""
    if not text:
        return 'untitled'
    # replace unicode quotes with ascii, remove markers
    s = text.strip()
    s = s.replace('“', '"').replace('”', '"')
    # Extract a reasonable short title within quotes if present
    m = re.search(r'"([^"]{2,200})"', s)
    if m:
        s = m.group(1)
    # normalize: keep ascii letters, numbers, dash and underscore
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", '_', s)
    s = s.strip('_')
    if not s:
        return 'untitled'
    return s[:max_len]


def parse_song_file(path: str) -> Tuple[str, str, Dict[str, str]]:
    """Parse a song text file with a title, lyrics body, and style sections.

    Expected format (flexible): title on the top lines, then the lyrics body,
    and sections labelled like 'SMALL MODEL STYLE', 'MEDIUM MODEL STYLE',
    'LARGE MODEL STYLE' (case-insensitive). Returns (title, lyrics, styles_dict).
    styles_dict keys are 'small','medium','large' (lowercase). If a style is
    missing, its value will be an empty string.
    """
    title = ''
    lyrics = ''
    styles = {'small': '', 'medium': '', 'large': ''}
    try:
        with open(path, 'r', encoding='utf8') as fh:
            text = fh.read()
    except Exception:
        return ('', '', styles)

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Extract optional leading metadata lines of the form KEY: value
    meta = {}
    lines_all = text.split('\n')
    i = 0
    while i < len(lines_all):
        line = lines_all[i].strip()
        if not line:
            i += 1
            break
        m = re.match(r'^([A-Za-z0-9_ -]+)\s*:\s*(.+)$', line)
        if m:
            k = m.group(1).strip().lower()
            v = m.group(2).strip()
            meta[k] = v
            i += 1
            continue
        else:
            break
    # Reconstruct text without leading metadata block
    text = '\n'.join(lines_all[i:])
    lines = [l for l in text.split('\n')]

    # Find title: prefer explicit metadata TITLE:, otherwise first non-empty line; extract quoted fragment if present
    if 'title' in meta and meta['title'].strip():
        title = meta['title'].strip()
    else:
        for L in lines:
            if L.strip():
                # clean common leading emoji marker
                candidate = L.strip()
                # extract quotes or fall back to trimmed line
                qm = re.search(r'["“”](.+?)["“”]', candidate)
                if qm:
                    title = qm.group(1).strip()
                else:
                    # remove leading song markers like 🎵
                    title = re.sub(r'^\W+', '', candidate)[:200].strip()
                break

    # Find the indices of style headings
    heading_pattern = re.compile(r'^(?:🎵\s*)?(SMALL|MEDIUM|LARGE)\s+MODEL\s+STYLE', re.I)
    current_section = 'lyrics'
    sections = {'lyrics': []}
    for line in lines:
        m = heading_pattern.match(line.strip())
        if m:
            size = m.group(1).lower()
            current_section = size
            sections[current_section] = []
            continue
        sections.setdefault(current_section, []).append(line)

    # Lyrics are everything before the first style heading (if any)
    lyrics = '\n'.join([l for l in sections.get('lyrics', [])]).strip()

    # Extract styles
    for k in ('small', 'medium', 'large'):
        styles[k] = '\n'.join(sections.get(k, [])).strip()

    # Attach parsed metadata into styles under special key '_meta' for backward compatibility
    if meta:
        styles['_meta'] = meta

    return (title, lyrics, styles)
