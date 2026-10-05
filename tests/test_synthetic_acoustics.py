"""Sanity check of the whole NVPA chain on SYNTHETIC audio (not a substitute for the real-data calibration).

Speech = harmonic tones; NV sounds are crude synthetic stand-ins (breath = band-limited noise, sniff = short
high-pass burst, laughter = amplitude-modulated voiced bursts, throat clearing = low-frequency burst) inserted in
the gaps between words. If the chain (alignment -> windows -> features -> detector -> matching) works, a detector
trained on such data must score well on fresh data, while the shuffle baseline must stay low.
"""
import numpy as np
import soundfile as sf

from nvtts_eval.core import ArtifactStore, run_metric
from nvtts_eval.data import Manifest, ManifestHeader, NVParser, Sample
from nvtts_eval.nvpa.detector import train_gap_detector
from nvtts_eval.nvpa.metric import NvpaMetric
from nvtts_eval.nvpa.summary import summarize_nvpa
from nvtts_eval.nvpa.training import build_training_set

SR = 16000
TYPES = ("laughter", "breathing", "sniff", "throatclearing")
META = dict(metric_version="1", input_hash="h", config_hash="c", config={})


def _noise(rng, n, lo, hi):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9)


def _nv(kind, rng):
    if kind == "breathing":
        n = int(SR * rng.uniform(0.4, 0.6)); x = _noise(rng, n, 200, 3500) * 0.06
        return x * np.hanning(n)
    if kind == "sniff":
        n = int(SR * rng.uniform(0.12, 0.2)); x = _noise(rng, n, 3000, 7500) * 0.12
        return x * np.exp(-np.linspace(0, 6, n))
    if kind == "laughter":
        n = int(SR * rng.uniform(0.5, 0.7)); t = np.arange(n) / SR
        carrier = sum(np.sin(2 * np.pi * 260 * k * t) / k for k in range(1, 4))
        return 0.1 * carrier * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 5.5 * t))) * np.hanning(n)
    n = int(SR * rng.uniform(0.22, 0.3)); t = np.arange(n) / SR
    return (0.12 * np.sin(2 * np.pi * 110 * t) + 0.1 * _noise(rng, n, 80, 500)) * np.hanning(n)


def _word(rng):
    n = int(SR * 0.25); t = np.arange(n) / SR; f0 = rng.uniform(100, 180)
    return 0.15 * sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 7)) * np.hanning(n)


def make_set(tmp_path, name, n_utts, n_speakers, seed):
    rng = np.random.default_rng(seed)
    gen = tmp_path / name
    gen.mkdir()
    samples, asr = [], []
    parser = NVParser()
    for u in range(n_utts):
        n_words = 12
        gaps = sorted(rng.choice(np.arange(1, n_words), size=rng.integers(2, 5), replace=False))
        kinds = {int(g): TYPES[rng.integers(0, 4)] for g in gaps}
        wav, words, stamps, t = [], [], [], 0.0
        wav.append(0.003 * rng.standard_normal(int(0.2 * SR))); t += 0.2
        for w in range(n_words):
            if w in kinds:
                nv = _nv(kinds[w], rng)
                wav.append(nv); t += len(nv) / SR
            stamps.append(round(t, 3)); words.append(f"w{w}")
            seg = _word(rng); wav.append(seg); t += len(seg) / SR
            pause = 0.03 + 0.04 * rng.random()
            wav.append(0.003 * rng.standard_normal(int(pause * SR))); t += pause
        x = np.concatenate(wav).astype("float32")
        text = " ".join((f"[{kinds[i]}] " if i in kinds else "") + w for i, w in enumerate(words))
        sid = f"spk_{u % n_speakers}_{u:03d}"
        sf.write(str(gen / f"{sid}.wav"), x, SR)
        samples.append(Sample.from_text(sid, f"spk_{u % n_speakers}", text, parser, generated_audio=f"{sid}.wav",
                                        ground_truth_audio=f"{name}/{sid}.wav"))
        asr.append({"sample_id": sid, "text": " ".join(w.upper() for w in words),
                    "tokens": [" " + w.upper() for w in words], "timestamps": stamps, "duration": len(x) / SR})
    hdr = ManifestHeader(track="A", source="ground_truth", split=name, audio_root=tmp_path.as_posix(),
                         generated_root=gen.as_posix())
    return Manifest(hdr, samples), asr


def test_detector_trained_on_synthetic_nv_generalises(tmp_path):
    train_m, train_asr = make_set(tmp_path, "train", 120, 8, seed=1)
    dev_m, dev_asr = make_set(tmp_path, "dev", 40, 5, seed=2)
    X, Y, groups, info = build_training_set(train_m, train_asr, TYPES)
    assert info["positives"]["breathing"] > 20 and info["invalid_windows"] == 0
    det = train_gap_detector(X, Y, groups, TYPES, max_iter=80)

    store = ArtifactStore(tmp_path / "run")
    store.write("asr", dev_asr, META)
    recs = run_metric(NvpaMetric(store, det), dev_m, store).records
    speaker_of = {s.sample_id: s.speaker_id for s in dev_m}
    s = summarize_nvpa(recs, speaker_of, tolerance=1, n_boot=100, shuffle_reps=30)
    print("\nsynthetic NVPA", round(s["value"], 3), "shuffle", round(s["shuffle_baseline"]["mean"], 3),
          {t: (round(v["mean"], 2), v["n"]) for t, v in s["by_type"].items()}, "spurious/100w", round(s["spurious_per_100_words"], 2))
    assert s["n_events"] > 80
    assert s["value"] > 0.8                                          # the chain can detect NVs when the signal is there
    assert s["shuffle_baseline"]["mean"] < 0.5 * s["value"]          # ... and placement matters
    assert all(v["mean"] > 0.6 for v in s["by_type"].values())
    assert s["reasons"].get("alignment_unreliable", 0) == 0
