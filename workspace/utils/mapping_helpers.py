import os
import json
import difflib
import torch


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
