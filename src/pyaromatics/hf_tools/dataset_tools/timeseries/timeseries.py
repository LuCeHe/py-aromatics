"""UEA classification, Informer-style forecasting, and Speech Commands loaders.

Table A (multivariate classification): FaceDetection, Heartbeat, PEMS-SF.
Table B (multivariate forecasting): electricity, traffic, weather.
Long natural audio: Google Speech Commands v0.02 (35-class or SC10).

Raw files live under ``$DATADIR/timeseries/``. ``train.py`` names are the aliases
in ``DATASET_ALIASES``. Run ``prepare_timeseries.py`` once to download.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import shutil
import tarfile
import wave
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from pyaromatics.stay_organized.utils import str2val

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

UEA_TABLE_A = ("FaceDetection", "Heartbeat", "PEMS-SF")
FORECAST_TABLE_B = ("electricity", "traffic", "weather")
SC10_LABELS = (
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
)

DATASET_ALIASES = {
    "facedetection": ("uea", "FaceDetection"),
    "uea_facedetection": ("uea", "FaceDetection"),
    "heartbeat": ("uea", "Heartbeat"),
    "uea_heartbeat": ("uea", "Heartbeat"),
    "pemssf": ("uea", "PEMS-SF"),
    "pems-sf": ("uea", "PEMS-SF"),
    "uea_pemssf": ("uea", "PEMS-SF"),
    "electricity": ("forecast", "electricity"),
    "ts_electricity": ("forecast", "electricity"),
    "traffic": ("forecast", "traffic"),
    "ts_traffic": ("forecast", "traffic"),
    "weather": ("forecast", "weather"),
    "ts_weather": ("forecast", "weather"),
    "speechcommands": ("speech", "speech_commands"),
    "speechcommands35": ("speech", "speech_commands"),
    "speechcommands10": ("speech", "speech_commands10"),
    "sc10": ("speech", "speech_commands10"),
}

_UEA_ZIP_URLS = {
    "FaceDetection": (
        "https://timeseriesclassification.com/aeon-toolkit/FaceDetection.zip",
        "http://www.timeseriesclassification.com/Downloads/FaceDetection.zip",
    ),
    "Heartbeat": (
        "https://timeseriesclassification.com/aeon-toolkit/Heartbeat.zip",
        "http://www.timeseriesclassification.com/Downloads/Heartbeat.zip",
    ),
    "PEMS-SF": (
        "https://timeseriesclassification.com/aeon-toolkit/PEMS-SF.zip",
        "http://www.timeseriesclassification.com/Downloads/PEMS-SF.zip",
    ),
}

_LAIGUOKUN = "https://github.com/laiguokun/multivariate-time-series-datasets/raw/master/datasets"
_FORECAST_URLS = {
    "electricity": (f"{_LAIGUOKUN}/electricity.txt.gz",),
    "traffic": (f"{_LAIGUOKUN}/traffic.txt.gz",),
    "weather": (
        # Informer/Autoformer 21-var weather, then Jena climate as a public fallback.
        "https://raw.githubusercontent.com/nlinhvu/Informer-tutorial/master/data/weather.csv",
        "https://github.com/nlinhvu/Informer-tutorial/raw/master/data/weather.csv",
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    ),
}

_SPEECH_COMMANDS_URLS = (
    "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
)

_UA = "Mozilla/5.0 (compatible; prepare_timeseries/1.0)"
_SC_RATE = 16_000


def is_timeseries_dataset_name(dataset_name: str) -> bool:
    return str(dataset_name).strip().lower() in DATASET_ALIASES


def list_timeseries_datasets() -> Dict[str, Tuple[str, str]]:
    return dict(DATASET_ALIASES)


def timeseries_root(datadir: Optional[str] = None, cachedir: Optional[str] = None) -> str:
    if datadir:
        return os.path.join(datadir, "timeseries")
    env = os.environ.get("PEBBLETRAIL_DATADIR")
    if env:
        return os.path.join(env, "timeseries")
    try:
        from thepebbletrail_official.paths import DATADIR
        return os.path.join(DATADIR, "timeseries")
    except Exception:
        pass
    if cachedir:
        parent = os.path.dirname(os.path.abspath(cachedir))
        return os.path.join(parent, "timeseries")
    return os.path.abspath(os.path.join(os.getcwd(), "data", "timeseries"))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: str, timeout: float = 120.0) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = Request(url, headers={"User-Agent": _UA})
    with urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp, dest)


def _download_first(urls: Sequence[str], dest: str) -> str:
    errors: List[str] = []
    for url in urls:
        try:
            print(f"  downloading {url}")
            _download(url, dest)
            return url
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"  failed: {exc}")
    raise RuntimeError("all download URLs failed:\n  " + "\n  ".join(errors))


def _looks_complete(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# UEA .ts parser + download
# ---------------------------------------------------------------------------

def _parse_uea_ts(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a UEA/UCR multivariate ``.ts`` file into ``(N, T, C)`` and labels."""
    data_started = False
    rows: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.lower().startswith("@data"):
                data_started = True
                continue
            if data_started:
                rows.append(s)
    if not rows:
        raise ValueError(f"no @data rows in {path}")

    series: List[np.ndarray] = []
    labels: List[str] = []
    for row in rows:
        parts = row.split(":")
        if len(parts) < 2:
            raise ValueError(f"expected dim:dim:...:label in {path}, got {row[:80]!r}")
        label = parts[-1]
        dim_strs = parts[:-1]
        dims = []
        for d in dim_strs:
            vals = [float("nan") if t in ("?", "") else float(t) for t in d.split(",")]
            dims.append(np.asarray(vals, dtype=np.float32))
        t_len = max(x.shape[0] for x in dims)
        arr = np.full((t_len, len(dims)), np.nan, dtype=np.float32)
        for c, x in enumerate(dims):
            arr[: x.shape[0], c] = x
        series.append(arr)
        labels.append(label)

    t_max = max(s.shape[0] for s in series)
    c_dim = series[0].shape[1]
    out = np.zeros((len(series), t_max, c_dim), dtype=np.float32)
    for i, s in enumerate(series):
        out[i, : s.shape[0]] = np.nan_to_num(s, nan=0.0)
    return out, np.asarray(labels)


def _encode_labels(train_y: np.ndarray, test_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    classes = sorted(set(train_y.tolist()) | set(test_y.tolist()), key=str)
    mapping = {c: i for i, c in enumerate(classes)}
    tr = np.asarray([mapping[x] for x in train_y], dtype=np.int64)
    te = np.asarray([mapping[x] for x in test_y], dtype=np.int64)
    return tr, te, [str(c) for c in classes]


def _uea_dir(root: str, name: str) -> str:
    return os.path.join(root, "uea", name)


def _uea_ready(root: str, name: str) -> bool:
    d = _uea_dir(root, name)
    return _looks_complete(os.path.join(d, "train.npz")) and _looks_complete(os.path.join(d, "test.npz"))


def _find_ts(extract_dir: str, split: str) -> str:
    split_u = split.upper()
    matches: List[str] = []
    for dirpath, _, files in os.walk(extract_dir):
        for fn in files:
            if fn.endswith(".ts") and split_u in fn.upper():
                matches.append(os.path.join(dirpath, fn))
    if not matches:
        raise FileNotFoundError(f"no *{split}*.ts under {extract_dir}")
    matches.sort(key=len)
    return matches[0]


def download_uea(root: str, name: str, force: bool = False) -> str:
    dest_dir = _uea_dir(root, name)
    if _uea_ready(root, name) and not force:
        print(f"[skip] UEA {name} already at {dest_dir}")
        return dest_dir
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f"{name}.zip")
    _download_first(_UEA_ZIP_URLS[name], zip_path)
    extract_dir = os.path.join(dest_dir, "_extract")
    if os.path.isdir(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    x_tr, y_tr_raw = _parse_uea_ts(_find_ts(extract_dir, "TRAIN"))
    x_te, y_te_raw = _parse_uea_ts(_find_ts(extract_dir, "TEST"))
    y_tr, y_te, classes = _encode_labels(y_tr_raw, y_te_raw)
    np.savez_compressed(os.path.join(dest_dir, "train.npz"), inputs=x_tr, labels=y_tr)
    np.savez_compressed(os.path.join(dest_dir, "test.npz"), inputs=x_te, labels=y_te)
    meta = {
        "name": name,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "seq_len": int(x_tr.shape[1]),
        "n_channels": int(x_tr.shape[2]),
        "n_classes": int(len(classes)),
        "classes": classes,
        "task": "classification",
    }
    with open(os.path.join(dest_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"[ok] UEA {name}: train {x_tr.shape} test {x_te.shape} classes {len(classes)}")
    return dest_dir


# ---------------------------------------------------------------------------
# Forecasting CSVs
# ---------------------------------------------------------------------------

def _forecast_csv_path(root: str, name: str) -> str:
    return os.path.join(root, "forecasting", f"{name}.csv")


def _save_numeric_csv(path: str, array: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ",".join(f"dim_{i}" for i in range(array.shape[1]))
    np.savetxt(path, array, delimiter=",", header=header, comments="")


def _load_numeric_table(path: str, gzipped: Optional[bool] = None) -> np.ndarray:
    use_gz = path.endswith(".gz") if gzipped is None else gzipped
    if use_gz:
        with gzip.open(path, "rt") as f:
            arr = np.loadtxt(f, ndmin=2)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return arr
    import csv
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"empty table {path}")
    start = 0
    try:
        [float(x) for x in rows[0] if str(x).strip() != ""]
    except ValueError:
        start = 1
    data: List[List[float]] = []
    for row in rows[start:]:
        vals: List[float] = []
        for x in row:
            try:
                vals.append(float(x))
            except ValueError:
                continue
        if vals:
            data.append(vals)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _jena_to_numeric(zip_path: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no csv in {zip_path}")
        with zf.open(names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            header = text.readline()
            cols = [c.strip().strip('"') for c in header.split(",")]
            keep = [i for i, c in enumerate(cols) if c.lower() not in ("date time", "date", "time")]
            rows = []
            for line in text:
                parts = line.strip().split(",")
                if len(parts) < len(cols):
                    continue
                try:
                    rows.append([float(parts[i]) for i in keep])
                except ValueError:
                    continue
    return np.asarray(rows, dtype=np.float32)


def download_forecast(root: str, name: str, force: bool = False) -> str:
    dest = _forecast_csv_path(root, name)
    if _looks_complete(dest) and not force:
        print(f"[skip] forecast {name} already at {dest}")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".download"
    url_used = _download_first(_FORECAST_URLS[name], tmp)
    if url_used.endswith(".zip") or "jena" in url_used.lower():
        arr = _jena_to_numeric(tmp)
        _save_numeric_csv(dest, arr)
        os.remove(tmp)
        print(f"[ok] weather from Jena climate fallback: {arr.shape} (not Autoformer 21-var)")
        return dest
    arr = _load_numeric_table(tmp, gzipped=url_used.endswith(".gz"))
    _save_numeric_csv(dest, arr)
    if os.path.isfile(tmp):
        os.remove(tmp)
    print(f"[ok] forecast {name}: {arr.shape} -> {dest}")
    return dest


def _window_forecast(
    series: np.ndarray,
    lookback: int,
    horizon: int,
    seed: int,
    val_frac_of_holdout: float = 0.5,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Informer 70/10/20 chronological split, then sliding windows."""
    n = int(series.shape[0])
    n_train = int(n * 0.7)
    n_test = int(n * 0.2)
    n_val = n - n_train - n_test
    borders = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, n),
    }
    mean = series[:n_train].mean(axis=0, keepdims=True)
    std = series[:n_train].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    series_n = (series - mean) / std

    def windows(lo: int, hi: int) -> Tuple[np.ndarray, np.ndarray]:
        # include lookback context from before the split start when possible
        start = max(0, lo - lookback)
        xs, ys = [], []
        last = hi - horizon
        t = start + lookback
        while t <= last:
            if t < lo:
                t += 1
                continue
            xs.append(series_n[t - lookback: t])
            ys.append(series_n[t: t + horizon])
            t += 1
        if not xs:
            raise ValueError(
                f"no forecasting windows in [{lo}, {hi}) with lookback={lookback} horizon={horizon}"
            )
        return np.stack(xs), np.stack(ys)

    splits = {}
    counts = {}
    for split, (lo, hi) in borders.items():
        x, y = windows(lo, hi)
        splits[split] = (x, y)
        counts[split] = int(x.shape[0])
    _ = seed  # kept for API stability with get_dataset shuffle
    _ = val_frac_of_holdout
    meta = {
        "n_channels": int(series.shape[1]),
        "lookback": int(lookback),
        "horizon": int(horizon),
        "seq_len": int(lookback),
        "n_train": counts["train"],
        "n_validation": counts["validation"],
        "n_test": counts["test"],
        "task": "forecasting",
    }
    return splits, meta


# ---------------------------------------------------------------------------
# Speech Commands
# ---------------------------------------------------------------------------

def _sc_raw_dir(root: str) -> str:
    return os.path.join(root, "speech_commands", "v0.02")


def _sc_hf_dir(root: str, subset: str) -> str:
    return os.path.join(root, "speech_commands", f"hf_{subset}")


def _read_wav_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        sw = w.getsampwidth()
        rate = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"expected 16-bit pcm, got sampwidth={sw} in {path}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    if rate != _SC_RATE:
        # linear resample to 16 kHz (clips are 1 s; keeps us dependency-free)
        t_old = np.linspace(0.0, 1.0, num=x.shape[0], endpoint=False)
        t_new = np.linspace(0.0, 1.0, num=int(round(x.shape[0] * _SC_RATE / max(rate, 1))), endpoint=False)
        x = np.interp(t_new, t_old, x).astype(np.float32)
    if x.shape[0] < _SC_RATE:
        x = np.pad(x, (0, _SC_RATE - x.shape[0]))
    elif x.shape[0] > _SC_RATE:
        x = x[:_SC_RATE]
    return x


def _sc_label_folders(raw_dir: str) -> List[str]:
    skip = {"_background_noise_"}
    names = []
    for fn in sorted(os.listdir(raw_dir)):
        p = os.path.join(raw_dir, fn)
        if os.path.isdir(p) and fn not in skip and not fn.startswith("."):
            names.append(fn)
    return names


def _load_list(path: str) -> set:
    if not os.path.isfile(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {ln.strip().replace("\\", "/") for ln in f if ln.strip()}


def download_speech_commands(root: str, force: bool = False) -> str:
    raw_dir = _sc_raw_dir(root)
    marker = os.path.join(raw_dir, ".extracted")
    if os.path.isfile(marker) and not force:
        print(f"[skip] Speech Commands already extracted at {raw_dir}")
        return raw_dir
    os.makedirs(os.path.dirname(raw_dir), exist_ok=True)
    tar_path = os.path.join(root, "speech_commands", "speech_commands_v0.02.tar.gz")
    _download_first(_SPEECH_COMMANDS_URLS, tar_path)
    if os.path.isdir(raw_dir) and force:
        shutil.rmtree(raw_dir)
    os.makedirs(raw_dir, exist_ok=True)
    print("  extracting Speech Commands (this can take a few minutes)...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(raw_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    print(f"[ok] Speech Commands extracted to {raw_dir}")
    return raw_dir


def _build_speech_index(root: str, subset: str, force: bool = False) -> str:
    """Write a path+label DatasetDict (wavs stay on disk)."""
    out_dir = _sc_hf_dir(root, subset)
    if os.path.isfile(os.path.join(out_dir, "dataset_dict.json")) and not force:
        print(f"[skip] Speech Commands index {out_dir}")
        return out_dir
    raw_dir = download_speech_commands(root, force=False)
    labels = list(SC10_LABELS) if subset == "speech_commands10" else _sc_label_folders(raw_dir)
    label_to_id = {n: i for i, n in enumerate(labels)}
    val_set = _load_list(os.path.join(raw_dir, "validation_list.txt"))
    test_set = _load_list(os.path.join(raw_dir, "testing_list.txt"))

    splits: Dict[str, Dict[str, List]] = {
        k: {"audio_path": [], "labels": []} for k in ("train", "validation", "test")
    }
    for lab in labels:
        folder = os.path.join(raw_dir, lab)
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if not fn.lower().endswith(".wav"):
                continue
            rel = f"{lab}/{fn}".replace("\\", "/")
            wav_path = os.path.join(folder, fn)
            if rel in test_set:
                dest = "test"
            elif rel in val_set:
                dest = "validation"
            else:
                dest = "train"
            splits[dest]["audio_path"].append(wav_path)
            splits[dest]["labels"].append(int(label_to_id[lab]))

    dsd = DatasetDict({k: Dataset.from_dict(v) for k, v in splits.items() if v["labels"]})
    os.makedirs(out_dir, exist_ok=True)
    dsd.save_to_disk(out_dir)
    meta = {
        "name": subset,
        "n_channels": 1,
        "seq_len": _SC_RATE,
        "n_classes": len(labels),
        "classes": labels,
        "task": "classification",
        "n_train": len(splits["train"]["labels"]),
        "n_validation": len(splits["validation"]["labels"]),
        "n_test": len(splits["test"]["labels"]),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[ok] Speech Commands {subset}: "
        f"train={meta['n_train']} val={meta['n_validation']} test={meta['n_test']}"
    )
    return out_dir


# ---------------------------------------------------------------------------
# Download orchestrator
# ---------------------------------------------------------------------------

def download_timeseries_datasets(
    datadir: str,
    names: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Download Table A + Table B + Speech Commands into ``$DATADIR/timeseries``."""
    root = timeseries_root(datadir=datadir)
    os.makedirs(root, exist_ok=True)
    wanted = set(names) if names else None
    done: Dict[str, str] = {}

    def want(kind: str, name: str) -> bool:
        if wanted is None:
            return True
        aliases = [k for k, (g, n) in DATASET_ALIASES.items() if g == kind and n == name]
        return name in wanted or any(a in wanted for a in aliases)

    for uea in UEA_TABLE_A:
        if want("uea", uea):
            done[uea] = download_uea(root, uea, force=force)
    for fc in FORECAST_TABLE_B:
        if want("forecast", fc):
            done[fc] = download_forecast(root, fc, force=force)
    if want("speech", "speech_commands") or want("speech", "speech_commands10") or wanted is None:
        download_speech_commands(root, force=force)
        done["speech_commands"] = _build_speech_index(root, "speech_commands", force=force)
        done["speech_commands10"] = _build_speech_index(root, "speech_commands10", force=force)

    ready_path = os.path.join(root, ".ready.json")
    with open(ready_path, "w", encoding="utf-8") as f:
        json.dump({"root": root, "datasets": sorted(done)}, f, indent=2)
    print(f"timeseries root: {root}")
    return done


# ---------------------------------------------------------------------------
# Loaders for get_dataset
# ---------------------------------------------------------------------------

def _npz_split_to_dataset(npz_path: str) -> Dataset:
    blob = np.load(npz_path)
    inputs = blob["inputs"].astype(np.float32)
    labels = blob["labels"].astype(np.int64)
    return Dataset.from_dict({
        "inputs": inputs.tolist(),
        "labels": labels.tolist(),
    })


def _classification_with_val(train: Dataset, test: Dataset, seed: int, val_frac: float = 0.1) -> DatasetDict:
    n = len(train)
    n_val = max(1, int(round(n * val_frac)))
    split = train.train_test_split(test_size=n_val, seed=seed, shuffle=True)
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": test,
    })


def _missing_msg(kind: str, name: str, root: str) -> str:
    return (
        f"Timeseries dataset {name!r} is not on disk at {root}. "
        f"Run: python prepare_timeseries.py"
    )


def get_timeseries_dataset(
    dataset_name: str,
    seed: int = 42,
    notes: str = "",
    cachedir: Optional[str] = None,
    datadir: Optional[str] = None,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    key = str(dataset_name).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown timeseries dataset {dataset_name!r}")
    kind, name = DATASET_ALIASES[key]
    root = timeseries_root(datadir=datadir, cachedir=cachedir)
    lookback = str2val(notes, "lookback", default=96, output_type=int)
    horizon = str2val(notes, "horizon", default=96, output_type=int)

    if kind == "uea":
        if not _uea_ready(root, name):
            raise FileNotFoundError(_missing_msg(kind, name, _uea_dir(root, name)))
        meta_path = os.path.join(_uea_dir(root, name), "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        train = _npz_split_to_dataset(os.path.join(_uea_dir(root, name), "train.npz"))
        test = _npz_split_to_dataset(os.path.join(_uea_dir(root, name), "test.npz"))
        dataset = _classification_with_val(train, test, seed=seed)
        n_channels = int(meta["n_channels"])
        seq_len = int(meta["seq_len"])
        n_outputs = int(meta["n_classes"])
        task = "classification"

    elif kind == "forecast":
        csv_path = _forecast_csv_path(root, name)
        if not _looks_complete(csv_path):
            raise FileNotFoundError(_missing_msg(kind, name, csv_path))
        series = _load_numeric_table(csv_path)
        splits, meta = _window_forecast(series, lookback=lookback, horizon=horizon, seed=seed)
        dataset = DatasetDict({
            split: Dataset.from_dict({
                "inputs": x.tolist(),
                "labels": y.tolist(),
            })
            for split, (x, y) in splits.items()
        })
        n_channels = int(meta["n_channels"])
        seq_len = int(lookback)
        n_outputs = int(horizon * n_channels)
        task = "forecasting"

    else:
        subset = name
        hf_dir = _sc_hf_dir(root, subset)
        if not os.path.isfile(os.path.join(hf_dir, "dataset_dict.json")):
            if os.path.isfile(os.path.join(_sc_raw_dir(root), ".extracted")):
                _build_speech_index(root, subset, force=False)
            else:
                raise FileNotFoundError(_missing_msg(kind, name, hf_dir))
        dataset = load_from_disk(hf_dir)
        meta_path = os.path.join(hf_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        n_channels = 1
        seq_len = int(meta.get("seq_len", _SC_RATE))
        n_outputs = int(meta["n_classes"])
        task = "classification"

    data_config = {
        "eval_strategy": "epoch",
        "eval_steps": 1,
        "neftune": None,
        "label_smoothing_factor": 0.0,
        "early_stopping_patience": 10 if task == "classification" else 5,
        "lr_scheduler_type": "constant",
        "max_seq_length": seq_len,
        "synthvocab": None,
        "modality": "timeseries",
        "task": task,
        "n_channels": n_channels,
        "n_outputs": n_outputs,
        "num_labels": n_outputs if task == "classification" else n_channels,
        "lookback": lookback,
        "horizon": horizon if task == "forecasting" else 0,
        "dataset_kind": kind,
        "dataset_canonical": name,
    }
    return dataset, data_config


# ---------------------------------------------------------------------------
# Collator, metrics, model wrapper (so train.py can stay a causal-LM trainer)
# ---------------------------------------------------------------------------

class TimeSeriesCollator:
    """Pad ``inputs`` to ``(B, T, C)`` and stack labels."""

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = float(pad_value)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        xs = []
        for ex in batch:
            if "audio_path" in ex and ex["audio_path"]:
                wav = _read_wav_mono(ex["audio_path"])
                xs.append(torch.as_tensor(wav, dtype=torch.float32).unsqueeze(-1))
            else:
                xs.append(torch.as_tensor(ex["inputs"], dtype=torch.float32))
        if xs[0].ndim == 1:
            xs = [x.unsqueeze(-1) for x in xs]
        max_t = max(x.shape[0] for x in xs)
        c = xs[0].shape[-1]
        inputs = xs[0].new_full((len(xs), max_t, c), self.pad_value)
        attention_mask = torch.zeros(len(xs), max_t, dtype=torch.long)
        for i, x in enumerate(xs):
            t = x.shape[0]
            inputs[i, :t] = x
            attention_mask[i, :t] = 1
        labels = [ex["labels"] for ex in batch]
        lab0 = torch.as_tensor(labels[0])
        if lab0.ndim == 0:
            labels_t = torch.as_tensor(labels, dtype=torch.long)
        else:
            labels_t = torch.stack([torch.as_tensor(y, dtype=torch.float32) for y in labels])
        return {
            "inputs": inputs,
            "attention_mask": attention_mask,
            "labels": labels_t,
        }


def timeseries_compute_metrics(data_config: Dict[str, Any]):
    task = data_config.get("task")

    def _cls(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        labels = np.asarray(labels)
        return {"accuracy": float((preds == labels).mean())}

    def _fc(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        mse = float(np.mean((preds - labels) ** 2))
        mae = float(np.mean(np.abs(preds - labels)))
        return {"mse": mse, "mae": mae}

    return _cls if task == "classification" else _fc


def wrap_causal_lm_for_timeseries(model, data_config: Dict[str, Any]):
    """Linear ``C -> d`` then the causal-LM backbone via ``inputs_embeds``."""
    import torch.nn as nn

    def inner_seq_model(backbone):
        inner = getattr(backbone, "model", None)
        if inner is not None and hasattr(inner, "forward") and not isinstance(inner, nn.Embedding):
            return inner
        if hasattr(backbone, "transformer"):
            return backbone.transformer
        return backbone

    class TimeSeriesWrapper(nn.Module):
        def __init__(
            self,
            backbone,
            n_channels: int,
            n_outputs: int,
            task: str,
            hidden_size: int,
            horizon: int = 0,
            n_target_channels: int = 0,
        ):
            super().__init__()
            self.backbone = backbone
            self.config = getattr(backbone, "config", None)
            self.task = task
            self.horizon = int(horizon)
            self.n_target_channels = int(n_target_channels) or n_outputs
            self.in_proj = nn.Linear(n_channels, hidden_size)
            if task == "classification":
                self.head = nn.Linear(hidden_size, n_outputs)
            else:
                self.head = nn.Linear(hidden_size, self.horizon * self.n_target_channels)
            self._inner = inner_seq_model(backbone)

        def gradient_checkpointing_enable(self, **kwargs):
            fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if callable(fn):
                fn(**kwargs)

        def gradient_checkpointing_disable(self):
            fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
            if callable(fn):
                fn()

        def num_parameters(self, only_trainable: bool = False, **kwargs) -> int:
            return sum(p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable))

        def forward(self, inputs=None, labels=None, attention_mask=None, input_ids=None, **kwargs):
            if inputs is None:
                raise ValueError("TimeSeriesWrapper expects float ``inputs`` of shape (B, T, C)")
            if inputs.ndim == 2:
                inputs = inputs.unsqueeze(-1)
            hidden = self.in_proj(inputs)
            inner_kwargs = {"inputs_embeds": hidden, "use_cache": False}
            if attention_mask is not None:
                inner_kwargs["attention_mask"] = attention_mask
            try:
                out = self._inner(**inner_kwargs)
            except TypeError:
                inner_kwargs.pop("use_cache", None)
                out = self._inner(**inner_kwargs)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            if attention_mask is None:
                pooled = h.mean(dim=1)
            else:
                mask = attention_mask.unsqueeze(-1).to(h.dtype)
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            loss = None
            if self.task == "classification":
                logits = self.head(pooled)
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits, labels.long())
            else:
                pred = self.head(pooled)
                b = pred.shape[0]
                logits = pred.view(b, self.horizon, self.n_target_channels)
                if labels is not None:
                    loss = nn.functional.mse_loss(logits, labels.float())
            return {"loss": loss, "logits": logits}

    hidden = int(
        getattr(model.config, "hidden_size", None)
        or getattr(model.config, "n_embd", None)
        or getattr(model.config, "d_model", 128)
    )
    task = data_config["task"]
    return TimeSeriesWrapper(
        backbone=model,
        n_channels=int(data_config["n_channels"]),
        n_outputs=int(data_config["n_outputs"]),
        task=task,
        hidden_size=hidden,
        horizon=int(data_config.get("horizon") or 0),
        n_target_channels=int(data_config.get("num_labels") or data_config.get("n_channels") or 1),
    )
