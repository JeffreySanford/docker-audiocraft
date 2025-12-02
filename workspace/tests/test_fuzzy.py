import json
import torch
from utils.mapping_helpers import find_fuzzy_match, compute_match_score


def test_find_exact_shape_match():
    sd = {
        'encoder.model.0.conv.conv.weight': torch.zeros((16, 3, 3, 3)),
        'encoder.model.1.conv.conv.weight': torch.zeros((32, 16, 3, 3)),
    }
    param_tensor = torch.zeros((16, 3, 3, 3))
    candidate = find_fuzzy_match('encoder.model.0.conv.conv.weight', param_tensor, sd)
    assert candidate == 'encoder.model.0.conv.conv.weight'


def test_compute_match_score_high_for_shape_and_suffix():
    sd = {
        'encoder.model.0.conv.conv.weight': torch.zeros((16, 3, 3, 3)),
    }
    name = 'encoder.model.0.conv.conv.weight'
    candidate = 'encoder.model.0.conv.conv.weight'
    score = compute_match_score(name, candidate, torch.zeros((16, 3, 3, 3)), sd)
    assert score >= 0.85


def test_find_fuzzy_match_for_similar_name():
    sd = {
        'encoder.0.conv.weight': torch.zeros((16, 3, 3, 3)),
    }
    param_tensor = torch.zeros((16, 3, 3, 3))
    candidate = find_fuzzy_match('encoder.model.0.conv.conv.weight', param_tensor, sd, ratio_thresh=0.5)
    assert candidate is not None

