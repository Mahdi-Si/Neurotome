"""A time-series model with a dataset, loadable by spec: `py:examples.signal_model:build`.

Run from the repo root (so `examples` is importable):

    neurotome capture py:examples.signal_model:build --sample rec01:0
    # or in the UI: model = py:examples.signal_model:build, Sample = Dataset, pick a segment

Besides `build()` (and the hooks of examples/custom_model.py) a module may define a dataset:
  list_samples() -> [{"ref", "label", "group"}]   the samples the UI offers (cheap: the model is not built)
  load_sample(ref) -> dict                         one sample as the model's own pipeline makes it:
      "input"  the model input (batch of 1)        "raw"    the signal before the transform (fs, t0, channels)
      "steps"  the transform, in words             "blocks" column groups of a (1, T, C) input, with labels
      "time"   {axis, step_s, t0}: input steps on the raw signal's clock   "info" metadata rows
  OUTPUT_KIND = "values"                           the output is a score per step, not logits: no softmax
  target_windows(index) -> [{"label", "start", "stop", "blocks", "kind"}]
                                                   the input steps output element `index` reads (Trace view bands)

The data are synthetic and deterministic: 4 recordings of a heart rate (bpm) and a pressure (mmHg) channel at
4 Hz, 30 min each, cut into three 10-min segments. The transform is framewise band power of both channels.
"""

import numpy as np
import torch
from torch import nn

OUTPUT_KIND = "values"

FS = 4.0  # Hz
SEG_S, SEGMENTS, RECORDINGS = 600, 3, 4
FRAME, HOP = 64, 32  # samples: 16 s windows every 8 s
BANDS = [(0.0625, 0.125), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]  # Hz, octaves up to Nyquist
CHANNELS = [("hr", "bpm"), ("pressure", "mmHg")]
STEPS = 1 + (int(SEG_S * FS) - FRAME) // HOP  # 74
INPUT_SHAPE = [1, STEPS, len(CHANNELS) * len(BANDS)]
# one label per output element (a score per step): the step and its frame centre, 8 + 8 i s from the segment start
CATEGORIES = [f"t{i} · +{(8 + 8 * i) // 60}:{(8 + 8 * i) % 60:02d}" for i in range(STEPS)]
CONTEXT, HORIZON = 12, 3  # steps: what a step's score may read of the pressure bands, and what it forecasts


def target_windows(index: int) -> list[dict]:
    """The input steps that matter for output element `index` (the Trace target): the step itself, the pressure
    frames before it (a contraction's rise), and the next steps it forecasts. Neurotome clips them to [0, T)."""
    return [
        {"label": f"step t{index}", "start": index, "stop": index + 1, "blocks": None, "kind": "target"},
        {"label": f"pressure context ({CONTEXT} steps)", "start": index - CONTEXT, "stop": index,
         "blocks": ["pressure · band power"], "kind": "context"},
        {"label": f"horizon t+1…t+{HORIZON}", "start": index + 1, "stop": index + 1 + HORIZON, "blocks": None,
         "kind": "horizon"},
    ]


def _recording(rec: int) -> np.ndarray:
    """(2, 30 min * FS) heart rate and pressure: contractions every ~3 min; odd recordings get a heart-rate dip
    after each contraction peak."""
    rng = np.random.default_rng(rec)
    n = int(SEG_S * SEGMENTS * FS)
    t = np.arange(n) / FS
    peaks = np.cumsum(rng.uniform(150, 210, 12))
    pressure = 10 + sum(45 * np.exp(-(((t - p) / 25) ** 2)) for p in peaks) + rng.normal(0, 1, n)
    noise = np.zeros(n)
    for i in range(1, n):  # AR(1): slow beat-to-beat variability
        noise[i] = 0.95 * noise[i - 1] + rng.normal(0, 0.8)
    hr = 132 + 3 * rec + 5 * np.sin(2 * np.pi * t / 300) + noise + 2 * np.sin(2 * np.pi * 0.3 * t)
    if rec % 2:
        hr -= sum(20 * np.exp(-(((t - p - 30) / 20) ** 2)) for p in peaks)
    return np.stack([hr, pressure]).astype(np.float32)


def _parse(ref: str) -> tuple[int, int]:
    rec, _, seg = ref.partition(":")
    if not (rec.startswith("rec") and rec[3:].isdigit() and seg.isdigit()):
        raise KeyError(f"unknown sample {ref!r}; refs look like 'rec01:0'")
    r, s = int(rec[3:]), int(seg)
    if not (1 <= r <= RECORDINGS and 0 <= s < SEGMENTS):
        raise KeyError(f"unknown sample {ref!r}: recordings rec01–rec{RECORDINGS:02d}, segments 0–{SEGMENTS - 1}")
    return r, s


def list_samples() -> list[dict]:
    return [
        {"ref": f"rec{r:02d}:{s}", "label": f"segment {s} · {s * SEG_S}–{(s + 1) * SEG_S} s", "group": f"rec{r:02d}"}
        for r in range(1, RECORDINGS + 1)
        for s in range(SEGMENTS)
    ]


def band_power(x: np.ndarray) -> np.ndarray:
    """(channels, samples) -> (steps, channels * bands): log10 band power of Hann-windowed frames, z-scored per
    feature over the segment."""
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME, axis=1)[:, ::HOP]  # (ch, steps, FRAME)
    frames = frames - frames.mean(-1, keepdims=True)
    spec = np.abs(np.fft.rfft(frames * np.hanning(FRAME), axis=-1)) ** 2
    freqs = np.fft.rfftfreq(FRAME, 1 / FS)
    bands = [spec[..., (freqs >= lo) & (freqs < hi if hi < FS / 2 else freqs <= hi)].sum(-1) for lo, hi in BANDS]
    feat = np.log10(np.stack(bands, -1) + 1e-6)  # (ch, steps, bands)
    feat = feat.transpose(1, 0, 2).reshape(feat.shape[1], -1)
    return (feat - feat.mean(0)) / (feat.std(0) + 1e-6)


def load_sample(ref: str) -> dict:
    r, s = _parse(ref)
    start = s * SEG_S
    raw = _recording(r)[:, int(start * FS) : int((start + SEG_S) * FS)]
    x = torch.from_numpy(band_power(raw).astype(np.float32)).unsqueeze(0)  # (1, 74, 10)
    labels = [f"{lo:g}–{hi:g} Hz" for lo, hi in BANDS]
    nb = len(BANDS)
    return {
        "input": x,
        "name": f"rec{r:02d} · segment {s}",
        "steps": [
            f"Cut segment {s} of rec{r:02d}: {start}–{start + SEG_S} s at {FS:g} Hz ({raw.shape[1]} samples per channel)",
            f"Frame: {FRAME / FS:g} s Hann windows ({FRAME} samples) every {HOP / FS:g} s → {x.shape[1]} steps",
            f"Band power per channel: |rFFT|² summed over {nb} octave bands {BANDS[0][0]:g}–{BANDS[-1][1]:g} Hz",
            "log10(power + 1e-6), then z-score each feature over the segment",
            f"Stack hr and pressure bands → (1, {x.shape[1]}, {x.shape[2]})",
        ],
        "info": {"recording": f"rec{r:02d}", "segment": s, "start_s": start, "duration_s": SEG_S, "fs_hz": FS,
                 "pattern": "dips after contractions" if r % 2 else "no dips"},
        "raw": {
            "fs": FS,
            "t0": float(start),
            "time_label": "s from recording start",
            "channels": [{"name": n, "unit": u, "values": raw[i]} for i, (n, u) in enumerate(CHANNELS)],
        },
        "blocks": [
            {"name": f"{n} · band power", "start": i * nb, "stop": (i + 1) * nb, "labels": labels, "unit": "z"}
            for i, (n, _) in enumerate(CHANNELS)
        ],
        # step i is the frame centred at start + (i * HOP + FRAME / 2) / FS
        "time": {"axis": 1, "step_s": HOP / FS, "t0": start + FRAME / 2 / FS},
    }


class SignalNet(nn.Module):
    """Conv over time, one self-attention layer, a per-step score head. Returns a dict, the score first."""

    def __init__(self, c: int = INPUT_SHAPE[-1], d: int = 16):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c, d, 5, padding=2), nn.GELU())
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):  # (1, T, C)
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=False)
        h = self.norm(h + a)
        return {"score": self.head(h).squeeze(-1), "attention": w}


def build() -> nn.Module:
    torch.manual_seed(0)
    return SignalNet()
