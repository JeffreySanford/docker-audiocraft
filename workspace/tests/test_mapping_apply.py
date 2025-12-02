import json
import torch
import torch.nn as nn
import os
from utils.mapping_helpers import apply_mapping_to_module


class DummyComp(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.l1 = nn.Linear(16, 16)
        self.encoder.l2 = nn.Linear(16, 32)

    def named_parameters(self, recurse=True):
        return [
            ('encoder.l1.weight', self.encoder.l1.weight),
            ('encoder.l1.bias', self.encoder.l1.bias),
            ('encoder.l2.weight', self.encoder.l2.weight),
            ('encoder.l2.bias', self.encoder.l2.bias),
        ]


def test_apply_mapping_basic(tmp_path):
    sd = {
        'encoder.l1.weight': torch.zeros((16, 16)),
        'encoder.l1.bias': torch.zeros((16,)),
        'encoder.l2.weight': torch.zeros((32, 16)),
        'encoder.l2.bias': torch.zeros((32,)),
    }
    comp = DummyComp()
    mapping = {
        'encoder.l1.weight': 'encoder.l1.weight',
        'encoder.l1.bias': 'encoder.l1.bias',
    }
    applied = apply_mapping_to_module(comp, sd, mapping, device='cpu')
    assert applied == 2
    # check param tensors assigned
    assert not comp.encoder.l1.weight.is_meta
    assert not comp.encoder.l1.bias.is_meta


def test_save_applied_mapping(tmp_path):
    sd = {
        'encoder.l1.weight': torch.zeros((16, 16)),
        'encoder.l1.bias': torch.zeros((16,)),
    }
    comp = DummyComp()
    mapping = {
        'encoder.l1.weight': 'encoder.l1.weight',
        'encoder.l1.bias': 'encoder.l1.bias',
    }
    applied = apply_mapping_to_module(comp, sd, mapping, device='cpu')
    assert applied == 2
    out_path = tmp_path / 'applied_map.json'
    simple_map = {k: mapping[k] for k in mapping}
    with open(out_path, 'w', encoding='utf8') as fh:
        json.dump(simple_map, fh)
    assert os.path.exists(out_path)
