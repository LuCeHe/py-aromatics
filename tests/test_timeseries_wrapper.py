"""TimeSeriesWrapper must be safetensors / HF-Trainer checkpoint safe."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
import torch.nn as nn

from pyaromatics.hf_tools.dataset_tools.timeseries.timeseries import (
    TimeSeriesCollator,
    _forecast_window_starts,
    _lazy_forecast_dataset,
    _window_forecast,
    reapply_timeseries_lazy_transform,
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


def test_lazy_forecast_eval_is_detected_as_timeseries():
    from pyaromatics.hf_tools.helpers_datasets import _eval_split_is_timeseries

    lookback, horizon, n_ch = 8, 12, 5
    series = np.arange(400 * n_ch, dtype=np.float32).reshape(400, n_ch)
    series_n, starts, _ = _window_forecast(series, lookback, horizon, seed=0)
    ds = _lazy_forecast_dataset(series_n, starts, lookback, horizon)
    assert ds["test"].column_names == ["t"]
    assert "inputs" not in ds["test"].column_names
    assert _eval_split_is_timeseries(ds, "test") is True
    assert _eval_split_is_timeseries(ds, "test", collator=TimeSeriesCollator()) is True
    assert _eval_split_is_timeseries(ds, "test", is_timeseries=True) is True


def test_forecast_windows_stay_lazy_after_shuffle():
    lookback, horizon, n_ch = 8, 12, 5
    series = np.arange(400 * n_ch, dtype=np.float32).reshape(400, n_ch)
    series_n, starts, meta = _window_forecast(series, lookback, horizon, seed=0)
    assert meta["n_train"] == int(starts["train"].shape[0])
    assert series_n.nbytes < 10_000
    ds = _lazy_forecast_dataset(series_n, starts, lookback, horizon)
    assert ds["train"].column_names == ["t"]
    t0 = int(starts["train"][0])
    ex = ds["train"][0]
    np.testing.assert_allclose(ex["inputs"], series_n[t0 - lookback: t0])
    np.testing.assert_allclose(ex["labels"], series_n[t0: t0 + horizon])
    ds["train"] = ds["train"].shuffle(seed=1)
    reapply_timeseries_lazy_transform(ds)
    ex1 = ds["train"][0]
    assert np.asarray(ex1["inputs"]).shape == (lookback, n_ch)
    assert np.asarray(ex1["labels"]).shape == (horizon, n_ch)
    batch = TimeSeriesCollator()([ds["train"][i] for i in range(3)])
    assert batch["inputs"].shape == (3, lookback, n_ch)
    assert batch["labels"].shape == (3, horizon, n_ch)


def test_forecast_window_starts_skip_pre_split_context():
    starts = _forecast_window_starts(lo=20, hi=50, lookback=8, horizon=10)
    assert int(starts[0]) == 20
    assert int(starts[-1]) == 40
