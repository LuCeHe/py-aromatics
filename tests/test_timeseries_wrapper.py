"""TimeSeriesWrapper must be safetensors / HF-Trainer checkpoint safe."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from pyaromatics.hf_tools.dataset_tools.timeseries.timeseries import (
    timeseries_compute_metrics,
    wrap_causal_lm_for_timeseries,
)


class _Cfg:
    hidden_size = 8


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, inputs_embeds=None, attention_mask=None, use_cache=None, **kwargs):
        return type("Out", (), {"last_hidden_state": inputs_embeds})()


class _CausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.model = _Inner()
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight


def _wrap():
    return wrap_causal_lm_for_timeseries(
        _CausalLM(),
        {"task": "classification", "n_channels": 2, "n_outputs": 3, "horizon": 0},
    )


def test_inner_is_alias_not_registered_module():
    model = _wrap()
    assert "_inner" not in model._modules
    assert model._inner is model.backbone.model
    assert not any(key.startswith("_inner.") for key in model.state_dict())


def test_state_dict_drops_tied_lm_head():
    model = _wrap()
    keys = set(model.state_dict())
    assert "backbone.model.embed_tokens.weight" in keys
    assert "backbone.lm_head.weight" not in keys


def test_safetensors_save_and_reload(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    model = _wrap()
    path = tmp_path / "model.safetensors"
    safetensors.save_file(model.state_dict(), str(path))
    loaded = _wrap()
    loaded.load_state_dict(safetensors.load_file(str(path)), strict=False)
    assert torch.equal(loaded.head.weight, model.head.weight)
    assert torch.equal(
        loaded.backbone.lm_head.weight,
        loaded.backbone.model.embed_tokens.weight,
    )


def test_save_pretrained_writes_safetensors(tmp_path):
    model = _wrap()
    dest = tmp_path / "export"
    model.save_pretrained(dest)
    assert (dest / "model.safetensors").is_file()


def test_wrapper_has_hf_trainer_load_best_attrs():
    model = _wrap()
    assert model._keys_to_ignore_on_save is None
    model.tie_weights()


def test_forward_classification():
    model = _wrap()
    out = model(inputs=torch.randn(2, 5, 2), labels=torch.tensor([0, 2]))
    assert out["logits"].shape == (2, 3)
    assert out["loss"] is not None


def test_forward_casts_float_inputs_to_bf16_backbone():
    backbone = _CausalLM()
    backbone = backbone.to(dtype=torch.bfloat16)
    model = wrap_causal_lm_for_timeseries(
        backbone,
        {"task": "classification", "n_channels": 2, "n_outputs": 3, "horizon": 0},
    )
    out = model(inputs=torch.randn(2, 5, 2), labels=torch.tensor([0, 2]))
    assert out["logits"].shape == (2, 3)
    assert out["loss"] is not None
    assert out["logits"].dtype == torch.bfloat16


def test_compute_metrics_accepts_compute_result():
    fn = timeseries_compute_metrics({"task": "classification"})
    assert "compute_result" in inspect.signature(fn).parameters
    logits = np.array([[10.0, 0.0], [0.0, 10.0]])
    labels = np.array([0, 1])
    assert fn((logits, labels))["accuracy"] == 1.0


def test_compute_metrics_batches_then_finalize():
    fn = timeseries_compute_metrics({"task": "classification"})
    # batch 1: 1/1 correct; batch 2: 0/1 correct → 0.5
    assert fn((np.array([[10.0, 0.0]]), np.array([0])), compute_result=False) == {}
    out = fn((np.array([[10.0, 0.0]]), np.array([1])), compute_result=True)
    assert out["accuracy"] == 0.5
    # next eval is independent
    assert fn((np.array([[0.0, 10.0]]), np.array([1])))["accuracy"] == 1.0


def test_compute_metrics_forecast_accumulates():
    fn = timeseries_compute_metrics({"task": "forecasting"})
    assert fn((np.array([1.0, 3.0]), np.array([0.0, 0.0])), compute_result=False) == {}
    out = fn((np.array([5.0]), np.array([2.0])), compute_result=True)
    assert out["mse"] == pytest.approx((1.0 + 9.0 + 9.0) / 3.0)
    assert out["mae"] == pytest.approx((1.0 + 3.0 + 3.0) / 3.0)
