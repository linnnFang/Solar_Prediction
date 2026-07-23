"""Phase 5-8 tests: adapters, models, baselines, integration, evaluator, notebook API."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.baseline_skippd.adapters import CNNForecastBatchAdapter, PVOnlyBatchAdapter
from src.baseline_skippd.models import PVOnlyMLP, SunsetForecastCNN
from src.baseline_skippd.baselines import NaivePersistence, ClearSkyModel, ClearSkyAdjustedPersistence, pv_window_frame
from src.baseline_skippd.integrations import VanillaTransformerBatchAdapter, build_transformer
from src.baseline_skippd.integrations.vanilla_transformer import train_transformer
from src.baseline_skippd.training import train_torch_model, predict_kw
from src.baseline_skippd.evaluator import evaluate, compute_metrics
from src.baseline_skippd import notebook_api as api
from src.vanilla_transformer.model import TransformerForecaster

REAL_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "SKIPPD"
CFG = api.load_skippd_config(None)


# ---------- Phase 5: adapters -------------------------------------------------
def test_cnn_adapter_shape(pipeline_dm):
    batch = api.get_sample_batch(pipeline_dm, "train")
    ins = CNNForecastBatchAdapter().to_inputs(batch)
    b = batch["images"].shape[0]
    assert ins["images_stacked"].shape == (b, 48, 64, 64)
    assert ins["pv_history"].shape == (b, 16)


def test_pv_only_adapter_shape(pipeline_dm):
    batch = api.get_sample_batch(pipeline_dm, "train")
    assert PVOnlyBatchAdapter().to_inputs(batch)["pv_history"].shape == (batch["images"].shape[0], 16)


def test_transformer_adapter_shape(pipeline_dm):
    batch = api.get_sample_batch(pipeline_dm, "train")
    x, y = VanillaTransformerBatchAdapter().to_xy(batch)
    b = batch["images"].shape[0]
    assert x.shape == (b, 16, 1) and y.shape == (b, 1)


def test_transformer_adapter_does_not_modify_source_batch(pipeline_dm):
    batch = api.get_sample_batch(pipeline_dm, "train")
    before = batch["pv_history"].clone()
    VanillaTransformerBatchAdapter().to_xy(batch)
    assert torch.equal(batch["pv_history"], before)


def test_transformer_forward_smoke(pipeline_dm):
    model = build_transformer(CFG, pipeline_dm)
    batch = api.get_sample_batch(pipeline_dm, "train")
    out = VanillaTransformerBatchAdapter().forward(model, batch)
    assert out.shape == (batch["images"].shape[0], 1)


def test_existing_transformer_backward_compatibility():
    # the untouched vanilla_transformer still works on a plain numeric sequence
    m = TransformerForecaster(n_features=3, context_length=8, horizon=1)
    assert m(torch.randn(4, 8, 3)).shape == (4, 1)


# ---------- Phase 6: models + baselines --------------------------------------
def test_pv_only_forward():
    assert PVOnlyMLP(pv_len=16)(torch.randn(5, 16)).shape == (5, 1)


def test_sunset_forward():
    assert SunsetForecastCNN()(torch.randn(2, 48, 64, 64), torch.randn(2, 16)).shape == (2, 1)


def test_parameter_count():
    m = SunsetForecastCNN()
    assert m.head[0].in_features == 48 * 16 * 16 + 16          # 12304
    assert sum(p.numel() for p in m.parameters()) > 13_000_000


def test_single_training_step(pipeline_dm):
    model = PVOnlyMLP(pv_len=16)
    cfg = api.load_skippd_config(None); cfg["trainer"]["max_epochs"] = 1; cfg["verbose"] = False
    _, hist = train_torch_model(model, pipeline_dm, cfg, PVOnlyBatchAdapter())
    assert len(hist["train_loss"]) == 1 and np.isfinite(hist["train_loss"][0])


def test_naive_persistence(pipeline_dm):
    frame = pv_window_frame(pipeline_dm.store, pipeline_dm.window_index, "test")
    pred = NaivePersistence().predict_kw(frame)
    assert np.array_equal(pred, frame["p_now_kw"])            # y_hat == power at t


def test_clear_sky_no_future_target(pipeline_dm):
    frame = pv_window_frame(pipeline_dm.store, pipeline_dm.window_index, "test")
    csm = ClearSkyModel.fit(pipeline_dm.store, pipeline_dm.split_manifest, pipeline_dm.window_index)
    csp = ClearSkyAdjustedPersistence(csm, 30.1)
    pred1 = csp.predict_kw(frame)
    frame["y_true_kw"] = frame["y_true_kw"] * 0 + 999.0       # corrupt the target
    pred2 = csp.predict_kw(frame)
    assert np.array_equal(pred1, pred2)                      # prediction never used the target


# ---------- Phase 7: evaluator -----------------------------------------------
def _fake_result(n=50, split="test"):
    rng = np.random.default_rng(0)
    t0 = np.datetime64("2017-06-24T14:00", "ns").astype("int64")
    issue = t0 + (np.arange(n) * 60_000_000_000)
    yt = rng.uniform(0, 30, n).astype(np.float32)
    return {"y_true_kw": yt, "y_pred_kw": (yt + rng.normal(0, 1, n)).astype(np.float32),
            "issue_time": issue, "target_time": issue + 900_000_000_000,
            "sample_index": np.arange(n), "split": split}


def test_metric_inverse_transform():
    # perfect prediction -> accuracy 1.0
    r = _fake_result(); r["y_pred_kw"] = r["y_true_kw"].copy()
    m = compute_metrics(r, capacity_kw=30.1)
    assert np.isclose(m["overall"]["acc_rmse"], 1.0) and np.isclose(m["overall"]["acc_mae"], 1.0)


def test_prediction_schema(tmp_path):
    from src.baseline_skippd.evaluator import predictions_frame
    df = predictions_frame(_fake_result(), "m")
    assert set(df.columns) == {"sample_index", "model_name", "issue_time", "target_time", "split",
                               "y_true_kw", "y_pred_kw", "absolute_error_kw", "squared_error_kw"}


def test_evaluator_same_samples_across_models(pipeline_dm):
    m1 = api.build_model("pv_only", CFG, pipeline_dm)
    m2 = api.build_model("vanilla_transformer", CFG, pipeline_dm)
    r1 = predict_kw(m1, pipeline_dm, CFG, m1.skippd["adapter"], "test")
    r2 = predict_kw(m2, pipeline_dm, CFG, m2.skippd["adapter"], "test")
    assert np.array_equal(r1["sample_index"], r2["sample_index"])
    assert np.array_equal(r1["issue_time"], r2["issue_time"])
    assert np.array_equal(r1["y_true_kw"], r2["y_true_kw"])   # identical targets/samples


# ---------- Phase 8: notebook API --------------------------------------------
@pytest.mark.skipif(not REAL_ROOT.exists(), reason="real SKIPP'D data not present")
def test_notebook_api_build_datamodule():
    dm = api.build_skippd_datamodule(CFG); dm.setup()
    assert len(dm.window_index) > 0 and dm.sample_schema()["images"]["shape"] == (16, 3, 64, 64)


def test_notebook_api_build_cnn(pipeline_dm):
    model = api.build_model("sunset_forecast", CFG, pipeline_dm)
    assert model.skippd["name"] == "sunset_forecast"
    assert isinstance(model.skippd["adapter"], CNNForecastBatchAdapter)


def test_notebook_api_build_existing_transformer(pipeline_dm):
    model = api.build_model("vanilla_transformer", CFG, pipeline_dm)
    assert isinstance(model, TransformerForecaster) and model.skippd["kind"] == "transformer"


def test_shared_test_sample_ids(pipeline_dm):
    # every model draws test samples from the one shared window index
    ids = pipeline_dm.window_index.subset("test").issue_time
    for name in ("pv_only", "sunset_forecast", "vanilla_transformer"):
        m = api.build_model(name, CFG, pipeline_dm)
        r = predict_kw(m, pipeline_dm, CFG, m.skippd["adapter"], "test")
        assert np.array_equal(r["issue_time"], ids)


def test_end_to_end_smoke_cpu(pipeline_dm, tmp_path):
    cfg = api.load_skippd_config(None)
    cfg["device"] = "cpu"; cfg["verbose"] = False
    cfg["trainer"]["max_epochs"] = 1; cfg["artifacts"]["dir"] = str(tmp_path)
    for name in ("pv_only", "vanilla_transformer"):
        model = api.build_model(name, cfg, pipeline_dm)
        api.train_model(model, pipeline_dm, cfg)
        metrics = api.evaluate_model(model, pipeline_dm, cfg)
        assert "acc_rmse" in metrics["overall"]
    naive = api.run_rule_baseline("naive_persistence", pipeline_dm, cfg)
    assert naive["overall"]["n"] > 0
