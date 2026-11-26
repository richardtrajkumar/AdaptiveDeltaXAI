"""Release-critical temporal-index and controller-age contracts for v1.9."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import core_v1_9 as core


class LastValue(torch.nn.Module):
    def predict_proba(self, x):
        return torch.sigmoid(x[:, -1, :1])


def config():
    return {"window": 2, "swing_lag": 1, "swing_steps": 8,
            "ema_alpha": .1, "score_k": 1e9, "drift_k": 1e9,
            "gamma": .9, "max_age": 2}


def test_synthetic_window_contract_is_next_observation():
    x=np.arange(12,dtype=np.float32).reshape(6,2); y=np.arange(6)
    xx, yy=core.windows(x,y,2)
    assert xx.shape == (4,2,2)
    assert xx[0].tolist() == x[0:2].tolist()
    assert yy.tolist() == y[2:].tolist()  # x[i-W:i] -> y[i]


def test_controller_records_endpoint_target_and_unlabelled_final_record():
    stream=np.linspace(-1,1,20,dtype=np.float32).reshape(10,2)
    records=core.run_adaptive(stream,LastValue(),config(),torch.device("cpu"),seed=7)
    assert records[0].t == 5  # Conservative 2W + delta + lag history.
    assert records[0].target_index == 6
    assert records[-1].target_index == len(stream)
    assert records[-1].target_valid is False


def test_cache_age_is_predecision_and_first_hit_is_one():
    stream=np.linspace(-1,1,20,dtype=np.float32).reshape(10,2)
    records=core.run_adaptive(stream,LastValue(),config(),torch.device("cpu"),seed=7)
    assert records[0].refreshed and records[0].age_before == 0 and records[0].age_after == 0
    first_hit=next(r for r in records[1:] if not r.refreshed)
    assert first_hit.age_before == 0 and first_hit.age_after == 1
