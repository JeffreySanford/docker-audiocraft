#!/usr/bin/env python3
import os
import sys
import time
import json
import torch
from omegaconf import OmegaConf
import threading
import time as _time
import torch.nn as nn

sys.path.append(os.path.dirname(__file__))

from utils import (
    find_fuzzy_match,
    compute_match_score,
    generate_transform_candidates,
    match_by_suffix,
    find_shape_match,
    apply_mapping_to_module,
)
from audiocraft.models.builders import get_lm_model, get_compression_model
from audiocraft.models.loaders import load_lm_model_ckpt, load_compression_model_ckpt
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from accelerate.big_modeling import init_empty_weights, dispatch_model, load_checkpoint_and_dispatch
from accelerate.utils.modeling import load_checkpoint_in_model, get_balanced_memory
import inspect
import argparse
import importlib.util
_utils_path = os.path.join(os.path.dirname(__file__), 'utils.py')
spec = importlib.util.spec_from_file_location('local_utils', _utils_path)
_local_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_local_utils)
parse_song_file = _local_utils.parse_song_file
slugify = _local_utils.slugify

# Small helper

def meta_param_stats(module):
    total = 0
    meta_count = 0
    meta_names = []
    for n, p in module.named_parameters(recurse=True):
        total += p.numel()
        if p.is_meta:
            meta_count += p.numel()
            meta_names.append(n)


    return total, meta_count, meta_names


def meta_buffer_stats(module):
    total = 0
    meta_count = 0
    meta_names = []
    for n, b in module.named_buffers(recurse=True):
        total += b.numel() if isinstance(b, torch.Tensor) else 0
        if isinstance(b, torch.Tensor) and b.is_meta:
            meta_count += b.numel()
            meta_names.append(n)
    return total, meta_count, meta_names


def assign_state_dict_to_module(module, sd, device='cpu'):
    replaced = 0
    sd_keys = set(sd.keys())
    module_params = dict(module.named_parameters())
    for name, _ in module_params.items():
        if name in sd_keys:
            v = sd[name]
            new_param = torch.nn.Parameter(v.to(device))
            # set into the module
            module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
            sub = module
            if module_name:
                for part in module_name.split('.'):
                    sub = getattr(sub, part)
            try:
                setattr(sub, param_name, new_param)
                replaced += 1
            except Exception:
                pass
    return replaced


def assign_buffers_from_state_dict(module, sd, device='cpu'):
    replaced = 0
    sd_keys = set(sd.keys())
    module_buffers = dict(module.named_buffers())
    for name in module_buffers.keys():
        if name in sd_keys:
            v = sd[name]
            # register or set buffer
            module_name, buf_name = name.rsplit('.', 1) if '.' in name else ('', name)
            sub = module
            if module_name:
                for part in module_name.split('.'):
                    sub = getattr(sub, part)
            try:
                # Use register_buffer to ensure it's treated as buffer and not param
                sub.register_buffer(buf_name, torch.as_tensor(v, device=device))
                replaced += 1
            except Exception:
                try:
                    setattr(sub, buf_name, torch.as_tensor(v, device=device))
                    replaced += 1
                except Exception:
                    pass
    return replaced


def apply_mapping_to_module(module, sd, mapping, device='cpu'):
    """Apply mapping dict {module_param_name -> sd_key} to module parameters.
    Returns the number of params applied.
    """
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


def write_debug_dump(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf8') as fh:
        for l in lines:
            fh.write(str(l) + '\n')


def match_by_suffix(name, sd_keys, max_suffix=6):
    parts = name.split('.')
    max_len = min(max_suffix, len(parts))
    for suffix_len in range(max_len, 0, -1):
        suffix = '.'.join(parts[-suffix_len:])
        for key in sd_keys:
            if key.endswith(suffix):
                return key
    return None


def generate_transform_candidates(name):
    # Produce a list of candidate transformed names to use for heuristic mapping
    candidates = set()
    candidates.add(name)
    # Common transforms
    candidates.add(name.replace('.model.', '.'))
    candidates.add(name.replace('encoder.', 'encoder.model.'))
    candidates.add(name.replace('decoder.', 'decoder.model.'))
    candidates.add(name.replace('.conv.conv.', '.conv.'))
    candidates.add(name.replace('.conv.', '.conv.conv.'))
    candidates.add(name.replace('module.', ''))
    candidates.add(name.replace('.weight', '.weight_g'))
    candidates.add(name.replace('.weight', '.weight_v'))
    # handle repeated fragments like conv.conv -> conv
    candidates.add(name.replace('.conv.conv.', '.conv.'))
    # attempt removing 'model' token
    candidates.add(name.replace('.model.', '.'))
    # heuristic: if there is a 'model' prefix inside, try removing it
    if '.model.' in name:
        candidates.add(name.replace('.model.', '.'))
    # remove index digits after tokens: 'encoder.model.0' -> 'encoder.model'
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p.isdigit():
            mutated = '.'.join(parts[:i] + parts[i+1:])
            candidates.add(mutated)
    # return in deterministic order
    return [c for c in sorted(candidates) if c]


def find_shape_match(name, param_tensor, sd):
    """Return a single state_dict key from sd where sd[key].shape == param_tensor.shape,
    or None if not unique or no match found.
   """
    candidates = []
    pshape = tuple(param_tensor.shape)
    for k, v in sd.items():
        if tuple(v.shape) == pshape:
            candidates.append(k)
    if len(candidates) == 1:
        return candidates[0]
    # If multiple candidates, prefer suffix match
    for c in candidates:
        if c.endswith(name.split('.')[-1]):
            return c
    # Tie-breaker: choose the candidate with longest common suffix
    best = None
    best_score = 0
    for c in candidates:
        # compute common suffix length
        rs = os.path.commonprefix(['.'.join(c.split('.')[::-1]), '.'.join(name.split('.')[::-1])])
        score = len(rs)
        if score > best_score:
            best_score = score
            best = c
    # only return 'best' if it is sensibly unique (score > 0)
    if best_score > 0:
        return best
    return None


def find_fuzzy_match(name, param_tensor, sd, top_n=5, ratio_thresh=0.75):
    """Use difflib to suggest the best name matches among sd keys where shape matches or where text similarity is high.
    Returns the candidate key if it meets thresholds; otherwise None.
    """
    import difflib
    keys = list(sd.keys())
    # prefer keys with same shape
    same_shape = [k for k, v in sd.items() if tuple(v.shape) == tuple(param_tensor.shape)]
    if same_shape:
        matches = difflib.get_close_matches(name, same_shape, n=top_n, cutoff=ratio_thresh)
        if matches:
            return matches[0]
    # fallback: try general name similarities
    matches = difflib.get_close_matches(name, keys, n=top_n, cutoff=ratio_thresh)
    if matches:
        return matches[0]
    # fallback to SequenceMatcher ratio search if no close matches but ratio above threshold
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


def normalize_name_variants(name):
    """Iteratively generate normalized variants of the name to help match differences in conv/tr conv naming.
    Returns a set of variants ordered by heuristics.
    """
    variants = set()
    variants.add(name)
    name0 = name
    # Apply several normalizations: replace repeated tokens, collapse 'convtr.convtr' etc.
    def add(v):
        if v and v not in variants:
            variants.add(v)
    # simple replacements
    add(name0.replace('.conv.conv.', '.conv.'))
    add(name0.replace('.convconv.', '.conv.'))
    add(name0.replace('.convtr.convtr.', '.convtr.'))
    add(name0.replace('.convtrconvtr.', '.convtr.'))
    add(name0.replace('.convtr', '.conv_tr'))
    add(name0.replace('.conv.tr', '.convtr'))
    add(name0.replace('.conv.', '.conv.conv.'))
    # handle encoder.model.X vs encoder.X
    add(name0.replace('.model.', '.'))
    add(name0.replace('encoder.model.', 'encoder.'))
    add(name0.replace('decoder.model.', 'decoder.'))
    add(name0.replace('.convconv', '.conv'))
    # Remove digits tokens in some places
    import re
    parts = name0.split('.')
    for i, p in enumerate(parts):
        if re.fullmatch(r"\d+", p):
            mutated = '.'.join(parts[:i] + parts[i+1:])
            add(mutated)
    # attempt swapping model ordering e.g., model.0.block -> block.0
    add(name0.replace('.model.', '.'))
    # return deterministic order
    return sorted(variants)


def compute_match_score(name, candidate, param_tensor, sd):
    """Compute a normalized (0..1) score for the fuzzy mapping candidate.
    Components:
    - shape match: +0.5 if shapes equal
    - suffix match: +0.25 if base param name suffix match
    - long common suffix length: 0.1 * (length_ratio)
    - difflib ratio scaled 0..0.15
    Normalized to 0..1
    """
    import difflib
    score = 0.0
    # shape match strong preference
    try:
        candidate_shape = tuple(sd[candidate].shape)
        param_shape = tuple(param_tensor.shape)
    except Exception:
        candidate_shape = None
        param_shape = tuple(param_tensor.shape)
    if candidate_shape is not None and candidate_shape == param_shape:
        score += 0.5
    # suffix match contribution if last token matches
    if candidate.endswith(name.split('.')[-1]):
        score += 0.2
    # compute longest common suffix token match as additional weight
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
    # longer suffixes count, scaled by depth
    score += min(0.1, 0.02 * cs_len)
    # difflib similarity scaled, modest contribution
    ratio = difflib.SequenceMatcher(None, name, candidate).ratio()
    score += 0.15 * ratio
    # clamp to 1.0
    if score > 1.0:
        score = 1.0
    return score


if __name__ == '__main__':
    # Monkey patch autocast/TorchAutocast to disable meta autocast
    try:
        import importlib
        _autocast_mod = importlib.import_module('audiocraft.utils.autocast')
        _orig_TorchAutocast = getattr(_autocast_mod, 'TorchAutocast', None)
        if _orig_TorchAutocast is not None:
            class _SafeTorchAutocast(_orig_TorchAutocast):
                def __init__(self, *args, **kwargs):
                    device_type = kwargs.get('device_type', None)
                    if device_type is None and len(args) >= 2:
                        device_type = args[1]
                    if device_type == 'meta':
                        kwargs['enabled'] = False
            pass
    except Exception:
        pass
    # Patch torch.autocast to allow device_type='meta' (no-op)
    try:
        _orig_autocast = torch.autocast
        from contextlib import nullcontext
        def _safe_autocast(device_type, dtype=None, enabled=True):
            if device_type == 'meta':
                return nullcontext()
            return _orig_autocast(device_type, dtype=dtype, enabled=enabled)
        torch.autocast = _safe_autocast
    except Exception:
        pass
    # Monkey-patch conditioners T5Conditioner.forward to avoid the 'embeds.to(self.output_proj.weight)' call
    try:
        cond_mod = importlib.import_module('audiocraft.modules.conditioners')
        if hasattr(cond_mod, 'T5Conditioner'):
            T5Conditioner = getattr(cond_mod, 'T5Conditioner')
            orig_forward = T5Conditioner.forward
            def _patched_forward(self, *args, **kwargs):
                placeholder_replaced = False
                orig_weight = None
                try:
                    # Debug: check if self.t5 has any meta params
                    try:
                        t5_meta_count = sum(1 for p in self.t5.parameters() if getattr(p, 'is_meta', False))
                        print('T5 meta param count at forward:', t5_meta_count)
                    except Exception:
                        pass
                    if hasattr(self, 'output_proj') and hasattr(self.output_proj, 'weight'):
                        w = self.output_proj.weight
                        if getattr(w, 'is_meta', False):
                            # create a cpu placeholder weight of same shape/dtype
                            try:
                                shape = tuple(w.shape)
                                cpu_w = torch.nn.Parameter(torch.zeros(shape, dtype=w.dtype, device='cpu'))
                                orig_weight = w
                                self.output_proj.weight = cpu_w
                                placeholder_replaced = True
                            except Exception:
                                placeholder_replaced = False
                    out = orig_forward(self, *args, **kwargs)
                    try:
                        # Out may be a tuple (embeds, mask) or BaseModelOutput; inspect
                        emb = None
                        if isinstance(out, tuple):
                            emb = out[0]
                        elif hasattr(out, 'last_hidden_state'):
                            emb = out.last_hidden_state
                        if emb is not None:
                            print('Embeds meta?:', getattr(emb, 'is_meta', False), 'dtype', getattr(emb, 'dtype', None), 'shape', getattr(emb, 'shape', None))
                    except Exception:
                        pass
                    return out
                finally:
                    if placeholder_replaced and orig_weight is not None:
                        self.output_proj.weight = orig_weight
            T5Conditioner.forward = _patched_forward
            print('Patched audiocraft.modules.conditioners.T5Conditioner.forward to avoid meta .to() device errors')
    except Exception:
        pass
    mname = 'facebook/musicgen-large'
    cache_dir = os.environ.get('AUDIOCRAFT_CACHE_DIR', '/workspace/cache')
    pkg_lm = load_lm_model_ckpt(mname, cache_dir=cache_dir)
    pkg_comp = load_compression_model_ckpt(mname, cache_dir=cache_dir)

    cfg_lm = OmegaConf.create(pkg_lm['xp.cfg'])
    cfg_comp = OmegaConf.create(pkg_comp['xp.cfg'])

    # Default to sys: meta skeleton for LM if environment/flags not set. Allow forcing LM to CPU
    cfg_lm.device = 'meta'
    if os.environ.get('FORCE_CPU_LM', '0').lower() in ('1', 'true', 'yes'):
        cfg_lm.device = 'cpu'
    cfg_comp.device = 'meta'

    parser = argparse.ArgumentParser(description='Run large MusicGen with offload and mapping helpers')
    parser.add_argument('--apply-mapping', type=str, default=None, help='Path to JSON mapping file to apply to compression params')
    parser.add_argument('--apply-proposed', action='store_true', help='Auto-apply high-confidence proposed fuzzy matches')
    parser.add_argument('--proposed-threshold', type=float, default=0.85, help='Score threshold in [0, 1] for auto-applying fuzzy proposals')
    parser.add_argument('--mapping-out', type=str, default='/workspace/debug/comp_applied_mapping.json', help='Path to save applied mapping (if any)')
    parser.add_argument('--apply-saved-mapping', action='store_true', help='Apply /workspace/debug/comp_applied_mapping.json mapping if present')
    parser.add_argument('--save-applied', action='store_true', help='Save applied mapping JSON to mapping-out when mappings are applied')
    parser.add_argument('--dry-run', action='store_true', help='Dry run: propose mappings but do not apply')
    parser.add_argument('--force-cpu-lm', action='store_true', help='Force LM to stay on CPU (do not dispatch to GPU)')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive prompts for fuzzy mapping application')
    parser.add_argument('--offload-dir', type=str, default='/workspace/offload', help='Directory to use for accelerate offloading')
    parser.add_argument('--max-gpu-memory', type=str, default=None, help='Override max GPU memory per device, eg "8GB" or "8GB,8GB"')
    parser.add_argument('--cpu-util-target', type=float, default=0.5, help='Target CPU utilization ratio in (0,1] used to choose num threads; default=0.5')
    parser.add_argument('--num-threads', type=int, default=None, help='Explicitly set torch.set_num_threads; overrides cpu-util-target')
    parser.add_argument('--warmup', action='store_true', help='Perform a brief CPU/GPU warmup after dispatch to exercise offloaded modules')
    parser.add_argument('--warmup-steps', type=int, default=3, help='Number of warmup iterations to run when --warmup is enabled')
    parser.add_argument('--monitor', action='store_true', help='Enable resource monitor during generation (samples CPU & GPU)')
    parser.add_argument('--monitor-interval', type=float, default=0.5, help='Monitor sampling interval in seconds')
    parser.add_argument('--stress', action='store_true', help='Run an optional CPU or GPU stress baseline (blocking) before generation')
    args = parser.parse_args()

    # Create skeletons
    # Temporarily no-op weight_norm to avoid Meta backend operator issues during skeleton creation
    orig_weight_norm = None
    try:
        orig_weight_norm = torch.nn.utils.weight_norm
        torch.nn.utils.weight_norm = lambda module, name='weight', dim=0: module
    except Exception:
        orig_weight_norm = None
    # Also no-op local module weight_norm if conv module imported a copy of it at import time
    try:
        _conv_mod = importlib.import_module('audiocraft.modules.conv')
        if hasattr(_conv_mod, 'weight_norm'):
            _conv_mod.weight_norm = lambda module, name='weight', dim=0: module
    except Exception:
        pass
    # If force CPU for LM, avoid creating LM with init_empty_weights so it's fully instantiated on CPU
    if cfg_lm.device == 'cpu':
        # Create compression skeleton as meta
        with init_empty_weights():
            comp = get_compression_model(cfg_comp)
        # Instantiate LM normally (on CPU)
        lm = get_lm_model(cfg_lm)
    else:
        # Both LM and compression instantiated as meta skeletons
        with init_empty_weights():
            lm = get_lm_model(cfg_lm)
            comp = get_compression_model(cfg_comp)
    # restore weight_norm
    if orig_weight_norm is not None:
        torch.nn.utils.weight_norm = orig_weight_norm

    total_lm, meta_lm, names_lm = meta_param_stats(lm)
    print('LM meta count BEFORE:', total_lm, meta_lm)
    total_lm_b, meta_lm_b, names_lm_b = meta_buffer_stats(lm)
    print('LM meta buffer count BEFORE:', total_lm_b, meta_lm_b)

    # Load and assign state
    sd_lm = pkg_lm.get('best_state')
    sd_comp = pkg_comp.get('best_state')

    if sd_lm:
        replaced_lm = assign_state_dict_to_module(lm, sd_lm, device='cpu')
        print('Replaced LM params count:', replaced_lm)
        # dump any remaining LM meta param names for debugging
        meta_lm_params = [n for n, p in lm.named_parameters() if getattr(p, 'is_meta', False)]
        if meta_lm_params:
            print('LM meta params sample (first 50):', meta_lm_params[:50])
            write_debug_dump('/workspace/debug/lm_meta_params.txt', meta_lm_params)

    if sd_comp:
        print('Compression state dict keys:', len(sd_comp))
        matched = 0
        comp_param_names = set(n for n, _ in comp.named_parameters())
        for k in sd_comp.keys():
            if k in comp_param_names:
                matched += 1
        print('Compression keys matched to comp parameters:', matched)
        replaced_comp = assign_state_dict_to_module(comp, sd_comp, device='cpu')
        # assign buffers too (e.g., quantizer codebooks) from compression state
        replaced_comp_bufs = assign_buffers_from_state_dict(comp, sd_comp, device='cpu')
        print('Replaced Comp buffers count:', replaced_comp_bufs)
        meta_comp_params = [n for n, p in comp.named_parameters() if getattr(p, 'is_meta', False)]
        if meta_comp_params:
            print('Comp meta params sample (first 50):', meta_comp_params[:50])
            write_debug_dump('/workspace/debug/comp_meta_params.txt', meta_comp_params)
        # If some compression params still meta, try to compute weight from weight_g/weight_v pairs
        comp_param_names = [n for n, _ in comp.named_parameters()]
        comp_param_set = set(comp_param_names)
        sd_keys = set(sd_comp.keys())
        missing = [n for n in comp_param_names if n not in sd_keys]
        # Try weight_g/weight_v -> weight reconstruction
        recomputed = 0
        for name in missing:
            if name.endswith('.weight'):
                base = name[:-len('.weight')]
                g_key = base + '.weight_g'
                v_key = base + '.weight_v'
                if g_key in sd_comp and v_key in sd_comp:
                    g = sd_comp[g_key]
                    v = sd_comp[v_key]
                    # compute norm across axes except first
                    try:
                        # handle zero/near-zero norms gracefully and support 2D and ND tensors
                        dims = tuple(range(1, v.ndim)) if v.ndim > 1 else (0,)
                        norms = v.norm(dim=dims, keepdim=True)
                        # if norms are zero, try sum of abs
                        if torch.any(norms == 0):
                            norms = v.abs().sum(dim=dims, keepdim=True)
                        g_shaped = g.reshape([g.shape[0]] + [1] * (v.ndim - 1)) if v.ndim > 1 else g
                        weight = v * (g_shaped / norms)
                        # assign computed weight into module param
                        module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
                        sub = comp
                        if module_name:
                            for part in module_name.split('.'):
                                sub = getattr(sub, part)
                        setattr(sub, param_name, torch.nn.Parameter(weight.to('cpu')))
                        recomputed += 1
                    except Exception:
                        pass
        if recomputed:
            print('Recomputed and assigned weights from weight_g/weight_v pairs for', recomputed, 'params')
        # Try to patch unmatched compression params by attempting key transformations
        comp_param_names = [n for n, _ in comp.named_parameters()]
        comp_param_set = set(comp_param_names)
        sd_keys = set(sd_comp.keys())
        missing = [n for n in comp_param_names if n not in sd_keys]
        print('Compression params missing from state dict (sample):', missing[:40])
        # Dump param names & state keys to debug files so we can inspect mismatches
        write_debug_dump('/workspace/debug/comp_param_names.txt', comp_param_names)
        write_debug_dump('/workspace/debug/comp_state_keys.txt', sorted(list(sd_keys)))
        # Apply mapping JSON early, if provided
        if args.apply_mapping:
            try:
                with open(args.apply_mapping, 'r', encoding='utf8') as fh:
                    mapping = json.load(fh)
                applied = apply_mapping_to_module(comp, sd_comp, mapping, device='cpu')
                print('Applied mapping file', args.apply_mapping, 'applied entries:', applied)
                if args.save_applied and applied > 0:
                    try:
                        os.makedirs(os.path.dirname(args.mapping_out), exist_ok=True)
                        # create simple mapping for saved file
                        simple_map = {k: (mapping[k]['candidate'] if isinstance(mapping[k], dict) else mapping[k]) for k in mapping if k in [n for n, _ in comp.named_parameters()]}
                        with open(args.mapping_out, 'w', encoding='utf8') as fh:
                            json.dump(simple_map, fh, indent=2)
                        print('Saved applied mapping to', args.mapping_out)
                    except Exception as e:
                        print('Failed to save applied mapping to', args.mapping_out, e)
            except Exception as e:
                print('Failed to apply mapping file', args.apply_mapping, e)
        # check for saved mapping and apply automatically if requested
        if args.apply_saved_mapping:
            try:
                if os.path.exists('/workspace/debug/comp_applied_mapping.json'):
                    with open('/workspace/debug/comp_applied_mapping.json', 'r', encoding='utf8') as fh:
                        mapping = json.load(fh)
                    applied = apply_mapping_to_module(comp, sd_comp, mapping, device='cpu')
                    print('Applied saved mapping', '/workspace/debug/comp_applied_mapping.json', 'applied entries:', applied)
                    if args.save_applied and applied > 0:
                        try:
                            os.makedirs(os.path.dirname(args.mapping_out), exist_ok=True)
                            simple_map = {k: (mapping[k]['candidate'] if isinstance(mapping[k], dict) else mapping[k]) for k in mapping if k in [n for n, _ in comp.named_parameters()]}
                            with open(args.mapping_out, 'w', encoding='utf8') as fh:
                                json.dump(simple_map, fh, indent=2)
                            print('Saved applied mapping to', args.mapping_out)
                        except Exception as e:
                            print('Failed to save applied mapping to', args.mapping_out, e)
            except Exception as e:
                print('Failed to apply saved mapping', e)
        # Try some heuristics to match keys (e.g., drop '.model' or convert 'encoder.model.' -> 'encoder.').
        heur_matches = 0
        for name in missing:
            # generate multiple transform candidates
            found = None
            # Use enhanced normalization and candidates generation
            candidates = generate_transform_candidates(name) + normalize_name_variants(name)
            for cand in candidates:
                if cand in sd_comp:
                    found = cand
                    break
            if found:
                # assign
                v = sd_comp[found]
                module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
                sub = comp
                if module_name:
                    for part in module_name.split('.'):
                        sub = getattr(sub, part)
                try:
                    setattr(sub, param_name, torch.nn.Parameter(v.to('cpu')))
                    heur_matches += 1
                except Exception:
                    pass
        if heur_matches:
            print('Heuristically matched and replaced compression params count:', heur_matches)
        print('Replaced Comp params count:', replaced_comp)
        # Try suffix-based matches and shape-based matches for remaining missing params
        suffix_matches = 0
        for name in missing:
            suffix_match = match_by_suffix(name, sd_keys)
            if suffix_match:
                try:
                    v = sd_comp[suffix_match]
                    module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
                    sub = comp
                    if module_name:
                        for part in module_name.split('.'):
                            sub = getattr(sub, part)
                    setattr(sub, param_name, torch.nn.Parameter(v.to('cpu')))
                    suffix_matches += 1
                except Exception:
                    pass
        if suffix_matches:
            print('Suffix matches assigned:', suffix_matches)

        # Try shape-based matches for remaining missing params
        sd_keys = set(sd_comp.keys())
        remaining = [n for n in comp_param_names if n not in sd_keys]
        shape_matches = 0
        for name in remaining[:]:
            # find param tensor to get shape
            module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
            sub = comp
            if module_name:
                for part in module_name.split('.'):
                    sub = getattr(sub, part)
            param_tensor = getattr(sub, param_name)
            match = find_shape_match(name, param_tensor, sd_comp)
            if match:
                try:
                    v = sd_comp[match]
                    setattr(sub, param_name, torch.nn.Parameter(v.to('cpu')))
                    shape_matches += 1
                    remaining.remove(name)
                except Exception:
                    pass
        if shape_matches:
            print('Shape-based matches assigned:', shape_matches)
        # For remaining missing parameters, attempt a fuzzy name match with safe ratio
        sd_keys = set(sd_comp.keys())
        remaining = [n for n in comp_param_names if n not in sd_keys]
        fuzzy_matches = 0
        proposed = {}
        for name in remaining[:]:
            module_name, param_name = name.rsplit('.', 1) if '.' in name else ('', name)
            sub = comp
            if module_name:
                for part in module_name.split('.'):
                    sub = getattr(sub, part)
            param_tensor = getattr(sub, param_name)
            candidate = find_fuzzy_match(name, param_tensor, sd_comp)
            if candidate:
                # compute score and keep high-confidence matches flagged for auto apply
                score = compute_match_score(name, candidate, param_tensor, sd_comp)
                proposed[name] = {'candidate': candidate, 'score': score}
                # Do not auto-apply fuzzy matches; log them and optionally prompt
        if proposed:
            # write proposals (including scores)
            os.makedirs('/workspace/debug', exist_ok=True)
            with open('/workspace/debug/comp_proposed_matches.json', 'w', encoding='utf8') as fh:
                json.dump(proposed, fh, indent=2)
            # also write a simple mapping to be used by --apply-mapping
            simple_map = {k: v['candidate'] for k, v in proposed.items()}
            with open('/workspace/debug/comp_proposed_matches_simple.json', 'w', encoding='utf8') as fh:
                json.dump(simple_map, fh, indent=2)
            print('Wrote proposed fuzzy matches to /workspace/debug/comp_proposed_matches.json')
            # If interactive mode, ask user to confirm and apply
            import sys
            # interactive check: CLI override or ENV
            if os.environ.get('INTERACTIVE', '0') == '1' or args.interactive or '--interactive' in sys.argv:
                print('Interactive mode: prompting to confirm fuzzy matches...')
                confirmed = []
                interactive_applied = {}
                for src, info in proposed.items():
                    cand = info['candidate']
                    score = info.get('score', 0.0)
                    yn = input(f"Apply fuzzy mapping {src} <- {cand}? score={score:.3f} [y/N]: ")
                    if yn.strip().lower() in ('y', 'yes'):
                        # assign
                        module_name, param_name = src.rsplit('.', 1) if '.' in src else ('', src)
                        sub = comp
                        if module_name:
                            for part in module_name.split('.'):
                                sub = getattr(sub, part)
                        try:
                            setattr(sub, param_name, torch.nn.Parameter(sd_comp[cand].to('cpu')))
                            confirmed.append((src, cand))
                            interactive_applied[src] = cand
                        except Exception as e:
                            print('Failed to assign interactive mapping', src, '->', cand, e)
                print('Confirmed fuzzy matches:', len(confirmed))
                if args.save_applied and len(interactive_applied) > 0:
                    try:
                        os.makedirs(os.path.dirname(args.mapping_out), exist_ok=True)
                        with open(args.mapping_out, 'w', encoding='utf8') as fh:
                            json.dump(interactive_applied, fh, indent=2)
                        print('Saved interactive applied mapping to', args.mapping_out)
                    except Exception as e:
                        print('Failed to save interactive mapping to', args.mapping_out, e)
            else:
                if os.environ.get('APPLY_PROPOSED', '0') == '1' or args.apply_proposed:
                    print('APPLY_PROPOSED set; applying proposed fuzzy matches automatically')
                    applied = 0
                    applied_map = {}
                    for src, info in proposed.items():
                        cand = info['candidate']
                        score = info.get('score', 0.0)
                        # Only auto-apply proposals >= 0.85
                        if score < args.proposed_threshold:
                            print(f"Skipping low-confidence auto-apply for {src} -> {cand} (score={score})")
                            continue
                        module_name, param_name = src.rsplit('.', 1) if '.' in src else ('', src)
                        sub = comp
                        if module_name:
                            for part in module_name.split('.'):
                                sub = getattr(sub, part)
                        try:
                            setattr(sub, param_name, torch.nn.Parameter(sd_comp[cand].to('cpu')))
                            applied += 1
                            applied_map[src] = cand
                        except Exception as e:
                            print('Failed to apply proposed match', src, '->', cand, e)
                    print('Applied proposed fuzzy matches:', applied)
                    if args.save_applied and applied > 0:
                        try:
                            os.makedirs(os.path.dirname(args.mapping_out), exist_ok=True)
                            with open(args.mapping_out, 'w', encoding='utf8') as fh:
                                json.dump(applied_map, fh, indent=2)
                            print('Saved applied mapping to', args.mapping_out)
                        except Exception as e:
                            print('Failed to save applied mapping to', args.mapping_out, e)
                else:
                    print('Not in interactive mode; not automatically applying fuzzy matches. Use INTERACTIVE=1 or APPLY_PROPOSED=1 to apply matches.')
        # Recompute & dump the final unmatched keys & counts
        sd_keys = set(sd_comp.keys())
        final_missing = [n for n in comp_param_names if n not in sd_keys]
        print('Final compression params STILL missing (sample 200):', len(final_missing), final_missing[:200])
        write_debug_dump('/workspace/debug/comp_unmatched_params.txt', final_missing)
        # JSON summary for quick inspection
        summary = {
            'compression_keys_total': len(sd_comp),
            'comp_params_total': len(comp_param_names),
            'initial_missing': len(missing),
            'recomputed': recomputed,
            'suffix_matches': suffix_matches,
            'heur_matches': heur_matches,
            'shape_matches': shape_matches,
            'final_missing_count': len(final_missing),
        }
        write_debug_dump('/workspace/debug/comp_match_summary.json', [json.dumps(summary)])
        # Write a more detailed dispatch report to help track module placements and memory map
        report_path = '/workspace/debug/dispatch_report.txt'
        try:
            with open(report_path, 'w', encoding='utf8') as fh:
                fh.write('Compression match summary:\n')
                fh.write(json.dumps(summary) + '\n')
                fh.write('Final missing params:\n')
                for m in final_missing:
                    fh.write(m + '\n')
                fh.write('\nModule device placement samples (LM):\n')
                for name, dev in module_devices(lm, 50):
                    fh.write(f'{name} -> {dev}\n')
                fh.write('\nModule device placement samples (Comp):\n')
                for name, dev in module_devices(comp, 50):
                    fh.write(f'{name} -> {dev}\n')
                fh.write('\nMax memory map used:\n')
                try:
                    fh.write(str(max_memory_map) + '\n')
                except Exception:
                    try:
                        fh.write(str(max_memory) + '\n')
                    except Exception:
                        fh.write('N/A\n')
                fh.write('\nDone report.\n')
            print('Wrote dispatch report to', report_path)
        except Exception as e:
            print('Failed to write dispatch report:', e)
        # If there are still unmatched params, attempt accelerate fallback to load checkpoint and dispatch
        sd_keys = set(sd_comp.keys())
        final_missing = [n for n in comp_param_names if n not in sd_keys]
        if len(final_missing) > 0:
            print('Some compression params remain unmatched; attempting accelerate fallback load...')
            comp_ckpt_dir = None
            # try to find compression_state_dict.bin in cache_dir
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    if f == 'compression_state_dict.bin' and mname.split('/')[-1] in root:
                        comp_ckpt_dir = os.path.join(root)
                        break
                if comp_ckpt_dir:
                    break
            if comp_ckpt_dir is not None:
                # copy into accelerate-friendly path
                comp_acc_dir = os.path.join(cache_dir, 'accelerate_ckpt_comp')
                os.makedirs(comp_acc_dir, exist_ok=True)
                import shutil
                shutil.copy(os.path.join(comp_ckpt_dir, 'compression_state_dict.bin'), os.path.join(comp_acc_dir, 'pytorch_model.bin'))
                print('Attempting load_checkpoint_and_dispatch for compression module using', comp_acc_dir)
                try:
                    # prefer load_checkpoint_and_dispatch where available; use introspection to pick compatible arguments
                    sig_dispatch = inspect.signature(dispatch_model)
                    sig_load = None
                    try:
                        sig_load = inspect.signature(load_checkpoint_and_dispatch)
                    except Exception:
                        sig_load = None
                    # compute local max_memory map
                    max_memory_local = {'cpu': '200GB'}
                    if torch.cuda.is_available():
                        total = torch.cuda.get_device_properties(0).total_memory
                        gb = int(total / (1024 ** 3))
                        max_memory_local[0] = f"{max(1, gb-1)}GB"
                    # Safe invocation wrapper
                    def invoke_load_checkpoint_and_dispatch(target_mod, ckpt_dir, device_map='auto', max_memory=None, offload_dir='/workspace/offload'):
                        # prefer the newer API if present
                        if sig_load is None:
                            return load_checkpoint_in_model(target_mod, ckpt_dir)
                        kwargs = {}
                        if 'device_map' in sig_load.parameters:
                            kwargs['device_map'] = device_map
                        if 'max_memory' in sig_load.parameters and max_memory is not None:
                            kwargs['max_memory'] = max_memory
                        # choose offload name based on signature
                        if 'offload_folder' in sig_load.parameters:
                            kwargs['offload_folder'] = offload_dir
                        elif 'offload_dir' in sig_load.parameters:
                            kwargs['offload_dir'] = offload_dir
                        # keep api call simple and guarded
                        return load_checkpoint_and_dispatch(target_mod, ckpt_dir, **kwargs)

                    invoke_load_checkpoint_and_dispatch(comp, comp_acc_dir, device_map='auto', max_memory=max_memory_local, offload_dir='/workspace/offload')
                except Exception as e:
                    print('load_checkpoint_and_dispatch for compression failed:', e)
                    try:
                        load_checkpoint_in_model(comp, comp_acc_dir)
                        print('load_checkpoint_in_model succeeded for compression')
                    except Exception as e2:
                        print('load_checkpoint_in_model for compression failed:', e2)
            else:
                print('Could not find compression state file in cache for accelerate fallback')

    total_lm, meta_lm, names_lm = meta_param_stats(lm)
    total_comp, meta_comp, names_comp = meta_param_stats(comp)
    total_lm_b, meta_lm_b, names_lm_b = meta_buffer_stats(lm)
    total_comp_b, meta_comp_b, names_comp_b = meta_buffer_stats(comp)
    print('LM meta count AFTER:', total_lm, meta_lm)
    print('LM meta buffer count AFTER:', total_lm_b, meta_lm_b)
    print('Comp meta count AFTER:', total_comp, meta_comp)
    print('Comp meta buffer count AFTER:', total_comp_b, meta_comp_b)
    if meta_lm_b:
        write_debug_dump('/workspace/debug/lm_meta_buffers.txt', names_lm_b)
    if meta_comp_b:
        write_debug_dump('/workspace/debug/comp_meta_buffers.txt', names_comp_b)

    # For debugging, check nested modules on the generated wrapper object instead of MusicGen
    # to avoid raising on wrappers that don't expose named_parameters directly

    # If LM meta is zero, dispatch LM to GPU 0
    max_memory = {'cpu': '200GB'}
    if torch.cuda.is_available():
        # compute default max memory map per GPU
        num_gpus = torch.cuda.device_count()
        gpu_gb = None
        for i in range(num_gpus):
            total = torch.cuda.get_device_properties(i).total_memory
            gb = int(total / (1024 ** 3))
            if args.max_gpu_memory:
                # support per device override like "7GB" or list "8GB,8GB"
                try:
                    if ',' in args.max_gpu_memory:
                        parts = args.max_gpu_memory.split(',')
                        if len(parts) == num_gpus:
                            max_memory[i] = parts[i]
                        else:
                            max_memory[i] = parts[0]
                    else:
                        max_memory[i] = args.max_gpu_memory
                except Exception:
                    max_memory[i] = f"{max(1, gb-1)}GB"
            else:
                max_memory[i] = f"{max(1, gb-1)}GB"

    if meta_lm == 0 and not args.force_cpu_lm:
        # Build explicit device map for all lm submodules -> GPU 0
        lm_devmap = {n: 0 for n, _ in lm.named_modules()}
        print('Dispatching LM with device_map size:', len(lm_devmap))
        import inspect
        sig = inspect.signature(dispatch_model)
        params = sig.parameters
        if 'offload_folder' in params:
            dispatch_model(lm, device_map=lm_devmap, offload_folder=args.offload_dir)
        elif 'offload_dir' in params:
            dispatch_model(lm, device_map=lm_devmap, offload_dir=args.offload_dir)
        else:
            dispatch_model(lm, device_map=lm_devmap)
    else:
        print('LM has meta params; cannot dispatch. meta count:', meta_lm)

    # For compression module, keep on CPU if not fully initialized; try to dispatch comp to CPU/disk mapping
    # Prefer to keep the compression module on CPU/disk (offload) but allow dispatching if desired/possible
    comp_devmap = {n: 'cpu' for n, _ in comp.named_modules()}
    print('Attempting to dispatch compression module to CPU/disk')
    sig = inspect.signature(dispatch_model)
    params = sig.parameters
    comp_dispatched = False
    force_cpu_comp = os.environ.get('FORCE_CPU_COMP', '0') == '1'
    try:
        if 'offload_folder' in params:
            dispatch_model(comp, device_map=comp_devmap, offload_folder=args.offload_dir)
        elif 'offload_dir' in params:
            dispatch_model(comp, device_map=comp_devmap, offload_dir=args.offload_dir)
        else:
            dispatch_model(comp, device_map=comp_devmap)
        comp_dispatched = True
    except Exception as e:
        print('Compression dispatch failed; keeping compression on CPU. Error:', e)
        comp_dispatched = False
    if force_cpu_comp:
        print('FORCE_CPU_COMP set; keeping compression on CPU even if dispatch might be possible')
        comp_dispatched = False

    # If compression is kept on CPU due to missing weights, ensure LM is on GPU
    if not comp_dispatched:
        print('LM is being dispatched to GPU; compression stays on CPU')
        try:
                if not args.dry_run:
                    # When comp is on CPU, dispatch LM only. For better memory balancing, wrap LM and compression and compute balanced mapping, preferring LM on GPU.
                    class _Wrapper(torch.nn.Module):
                        def __init__(self, lm_mod, comp_mod):
                            super().__init__()
                            self.lm = lm_mod
                            self.compression_model = comp_mod
                    wrapper = _Wrapper(lm, comp)
                    # Attempt to compute balanced memory mapping across available GPUs
                    max_memory_map = {'cpu': '200GB'}
                    try:
                        if torch.cuda.is_available():
                            total = torch.cuda.get_device_properties(0).total_memory
                            gb = int(total / (1024 ** 3))
                            max_gpu = max(1, gb - 1)
                            num_gpus = torch.cuda.device_count()
                            for i in range(num_gpus):
                                max_memory_map[i] = f"{max_gpu}GB"
                            print('Computed max_memory_map:', max_memory_map)
                            # For larger models, prefer accelerate's get_balanced_memory if available
                            try:
                                max_memory_map = get_balanced_memory(wrapper, max_memory_map)
                                print('get_balanced_memory returned a map of size', len(max_memory_map))
                            except Exception:
                                print('get_balanced_memory not available or failed; using local max_memory_map')
                                pass
                    except Exception as e:
                        print('Could not compute balanced memory map:', e)
                    # Build device map for LM-only assignment to GPU 0 for performance
                    lm_devmap = {n: 0 for n, _ in lm.named_modules()}
                    # Use wrapper-based dispatch so accelerate can fragment/replicate across GPUs or offload as needed
                    try:
                        # prefer to pass max_memory if dispatch signature supports it
                        kw = {}
                        if 'max_memory' in params:
                            kw['max_memory'] = max_memory_map
                        if 'offload_folder' in params:
                            kw['offload_folder'] = args.offload_dir
                        elif 'offload_dir' in params:
                            kw['offload_dir'] = args.offload_dir
                        dispatch_model(wrapper, device_map=lm_devmap, **kw)
                    except Exception as e:
                        print('Wrapper dispatch failed; trying to dispatch LM alone: ', e)
                        # fallback to LM single module dispatch
                        if 'offload_folder' in params:
                            dispatch_model(lm, device_map=lm_devmap, offload_folder='/workspace/offload')
                        elif 'offload_dir' in params:
                            dispatch_model(lm, device_map=lm_devmap, offload_dir='/workspace/offload')
                        else:
                            dispatch_model(lm, device_map=lm_devmap)
        except Exception as e:
            print('Explicit LM dispatch to GPU failed:', e)
        if args.force_cpu_lm:
            print('force_cpu_lm set; not dispatching LM to GPU')

    # mapping already applied earlier in the compression assignment section

    model = MusicGen(mname, comp, lm, max_duration=60)
    # Compute generation duration: prefer explicit metadata DURATION, otherwise
    # use lyrics length estimate or 30s, whichever is more
    def estimate_duration_from_text(text):
        # Estimate seconds from number of words. Typical sung words/sec ~2.5
        words = len(text.split())
        est = int(round(words / 2.5)) if words > 0 else 0
        return max(30, est)

    # Default duration
    gen_duration = 30
    # If a song file exists, parse it to obtain title/lyrics/styles
    prompt_file = '/workspace/lyrics_terraform_my_heart.txt'
    prompt_text = None
    title = None
    styles = None
    # Always prefer a fixed 60s generation for now (ignore metadata DURATION)
    if os.path.exists(prompt_file):
        try:
            title, lyrics, styles = parse_song_file(prompt_file)
            gen_duration = 60
        except Exception:
            gen_duration = 60
    model.set_generation_params(duration=gen_duration)

    # Helper: materialize any meta parameters/buffers in a module (replace with zeros on CPU)
    def materialize_meta_tensors(module, device='cpu'):
        # Parameters
        for name, p in list(module.named_parameters(recurse=True)):
            try:
                if getattr(p, 'is_meta', False):
                    shape = tuple(p.shape)
                    dtype = p.dtype if hasattr(p, 'dtype') else torch.float32
                    new_p = torch.nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))
                    parent, attr = (name.rsplit('.', 1) + [''])[:2] if '.' in name else ('', name)
                    sub = module
                    if parent:
                        for part in parent.split('.'):
                            sub = getattr(sub, part)
                    try:
                        setattr(sub, attr, new_p)
                    except Exception:
                        pass
            except Exception:
                pass
        # Buffers
        for name, b in list(module.named_buffers(recurse=True)):
            try:
                if isinstance(b, torch.Tensor) and getattr(b, 'is_meta', False):
                    shape = tuple(b.shape)
                    dtype = b.dtype if hasattr(b, 'dtype') else torch.float32
                    parent, attr = (name.rsplit('.', 1) + [''])[:2] if '.' in name else ('', name)
                    sub = module
                    if parent:
                        for part in parent.split('.'):
                            sub = getattr(sub, part)
                    try:
                        sub.register_buffer(attr, torch.zeros(shape, dtype=dtype, device=device))
                    except Exception:
                        try:
                            setattr(sub, attr, torch.zeros(shape, dtype=dtype, device=device))
                        except Exception:
                            pass
            except Exception:
                pass


    # Utility: report where modules are placed (device of parameters/buffers) for debugging
    def module_devices(module, max_print=50):
        out = []
        for name, sub in module.named_modules():
            # skip the root
            if name == '':
                continue
            try:
                # infer device by first parameter or buffer
                dev = None
                for _, p in sub.named_parameters(recurse=False):
                    dev = p.device
                    break
                if dev is None:
                    for _, b in sub.named_buffers(recurse=False):
                        if isinstance(b, torch.Tensor):
                            dev = b.device
                            break
                out.append((name, str(dev)))
            except Exception:
                out.append((name, 'unknown'))
            if len(out) >= max_print:
                break
        return out

    # If the user provided --num-threads or a CPU utilization target, configure torch threads early
    try:
        if args.num_threads is not None:
            num_threads = args.num_threads
        else:
            # compute from cpu-util-target
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            target = min(max(0.05, float(args.cpu_util_target)), 1.0)
            num_threads = max(1, int(cpu_count * target))
        torch.set_num_threads(num_threads)
        print('Set torch.num_threads to', num_threads)
    except Exception as e:
        print('Failed to set torch threads:', e)

    if args.stress:
        print('Starting stress baseline.. (Ctrl-C to exit)')
        try:
            # stress CPU or GPU depending on availability
            if torch.cuda.is_available() and not args.force_cpu_lm:
                # keep small random workloads on GPU
                import time
                while True:
                    t = torch.randn(4096, 4096, device='cuda')
                    _ = t @ t.T
                    time.sleep(0.1)
            else:
                import time
                while True:
                    a = torch.randn(2000, 200)
                    b = torch.randn(200, 2000)
                    _ = a @ b
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print('Stress stopped by user')
    if not args.dry_run:
        # run a quick generation
        print('Starting generation...')
        monitor = None
        if args.monitor:
            monitor = ResourceMonitor(out_path='/workspace/debug/monitor.log', interval=args.monitor_interval)
            try:
                monitor.start()
                print('Started resource monitor: /workspace/debug/monitor.log')
            except Exception as e:
                print('Failed to start monitor', e)
        # Optionally do a short warmup to load and exercise CPU/GPU resources
        if args.warmup:
            print('Running warmup:', args.warmup_steps, 'iterations')
            for i in range(args.warmup_steps):
                try:
                    # If LM on GPU, run quick forward to push it. Otherwise do CPU ops
                    if not args.force_cpu_lm and torch.cuda.is_available():
                        # run a tiny tensor pass on GPU
                        t = torch.randn(4096, 4096, device='cuda')
                        _ = t @ t.T
                    else:
                        a = torch.randn(2000, 200)
                        b = torch.randn(200, 2000)
                        _ = a @ b
                except Exception as e:
                    print('Warmup step failed:', e)
        # Print out sample device placement info for LM and compression modules
        print('LM module device placements (sample):')
        for name, dev in module_devices(lm, max_print=30):
            print(' ', name, '->', dev)
        print('Compression module device placements (sample):')
        for name, dev in module_devices(comp, max_print=30):
            print(' ', name, '->', dev)
        start = time.time()
        try:
            with torch.no_grad():
                        # Build prompt_text from parsed song file if available
                        if prompt_text is None:
                            # attempt to parse again if earlier parse didn't happen
                            if os.path.exists(prompt_file):
                                try:
                                    title, lyrics, styles = parse_song_file(prompt_file)
                                except Exception:
                                    title, lyrics, styles = (None, None, None)
                        if styles and isinstance(styles, dict):
                            style_section = styles.get('large', '')
                        else:
                            style_section = ''
                        if lyrics and lyrics.strip():
                            prompt_text = lyrics.strip()
                        else:
                            prompt_text = 'a rock ballad with guitars, drums and bass riffs'
                        if style_section:
                            prompt_text = prompt_text + '\n\nStyle: ' + style_section

                        # Materialize any remaining meta tensors in compression and relevant conditioner submodules
                        try:
                            # Target compression module first
                            materialize_meta_tensors(comp, device='cpu')
                        except Exception:
                            pass
                        try:
                            # If LM has a condition provider, materialize its conditioners
                            if hasattr(lm, 'condition_provider') and lm.condition_provider is not None:
                                materialize_meta_tensors(lm.condition_provider, device='cpu')
                        except Exception:
                            pass

                        wavs = model.generate([prompt_text])
        finally:
            if monitor:
                monitor.stop()
        print('Gen done in', time.time() - start)
        # Choose output filename based on title (if available)
        if title and title.strip():
            slug = slugify(title)
            out_path = f'/workspace/output/{slug}_large.wav'
        else:
            out_path = '/workspace/output/out_large_with_lyrics.wav'
        audio_write(out_path, wavs[0], model.sample_rate, strategy='loudness')
        print('Saved', out_path)
    else:
        print('Dry-run mode: skipping actual generation')



