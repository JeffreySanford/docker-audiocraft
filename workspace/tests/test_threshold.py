from utils.mapping_helpers import compute_match_score, find_fuzzy_match
import torch


def test_score_above_threshold_for_exact():
    sd = {
        'encoder.0.conv.weight': torch.zeros((16,3,3,3)),
    }
    name = 'encoder.model.0.conv.conv.weight'
    candidate = 'encoder.0.conv.weight'
    score = compute_match_score(name, candidate, torch.zeros((16,3,3,3)), sd)
    assert score >= 0.5


def test_fuzzy_match_succeeds_with_lower_ratio():
    sd = {
        'encoder.0.conv.weight': torch.zeros((16,3,3,3)),
    }
    name = 'encoder.model.0.conv.conv.weight'
    res = find_fuzzy_match(name, torch.zeros((16,3,3,3)), sd, ratio_thresh=0.4)
    assert res is not None
