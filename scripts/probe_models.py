"""Smoke-test the three real model backends on a few manifest samples (run this on YOUR machine).

    python scripts\\probe_models.py --manifest manifests\\dev_gt_A.jsonl --config configs\\default.yaml --n 3

It prints, per backend, exactly the facts the evaluator relies on and that could not be
verified without the weights:
  ASR     : transcript vs reference, whether token timestamps exist (needed by NVPA), speed
  DNSMOS  : sig / bak / ovrl / p808, including on the LONGEST clip (window handling)
  ECAPA   : embedding size, and cosine for same-speaker vs different-speaker pairs
            (same-speaker should be clearly higher)
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nvtts_eval.audio import load_mono  # noqa: E402
from nvtts_eval.cli import build_metrics  # noqa: E402
from nvtts_eval.config import load_config  # noqa: E402
from nvtts_eval.data.manifest import Manifest  # noqa: E402
from nvtts_eval.metrics.wer import edit_counts, normalize_text  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--skip", nargs="*", default=[], choices=["asr", "pmos", "ss"])
    a = ap.parse_args()

    cfg = load_config(a.config)
    m = Manifest.load(a.manifest)
    samples = m.samples[: a.n]
    paths = {s.sample_id: m.resolve_generated(s) for s in m.samples}
    longest = max(m.samples, key=lambda s: Path(paths[s.sample_id]).stat().st_size)
    metrics = build_metrics(cfg, [x for x in ("asr", "pmos", "ss") if x not in a.skip])

    if "asr" in metrics:
        print("=== ASR ===")
        be = metrics["asr"].backend
        be.setup()
        for s in samples:
            wav, sr = load_mono(paths[s.sample_id], 16000)
            t0 = time.time()
            r = be.transcribe(wav, sr)
            dt = time.time() - t0
            ref, hyp = normalize_text(s.clean_text).split(), normalize_text(r.text).split()
            sub, dele, ins = edit_counts(ref, hyp)
            print(f"{s.sample_id}: {len(wav)/sr:.1f}s audio, {dt:.2f}s decode, WER={(sub+dele+ins)/max(len(ref),1):.3f}")
            print(f"  REF: {' '.join(ref)[:160]}\n  HYP: {' '.join(hyp)[:160]}")
            print(f"  tokens={'yes' if r.tokens else 'NO'} timestamps={'yes' if r.timestamps else 'NO'}"
                  + (f" (first: {list(zip(r.tokens[:5], [round(t,2) for t in r.timestamps[:5]]))})" if r.tokens and r.timestamps else ""))

    if "pmos" in metrics:
        print("\n=== DNSMOS ===")
        be = metrics["pmos"].backend
        for s in samples + ([longest] if longest not in samples else []):
            wav, sr = load_mono(paths[s.sample_id], 16000)
            print(f"{s.sample_id} ({len(wav)/sr:.1f}s):", {k: round(v, 3) for k, v in be.score(wav, sr).items()})

    if "ss" in metrics:
        print("\n=== ECAPA ===")
        be = metrics["ss"].backend
        by_spk = {}
        for s in m.samples:
            by_spk.setdefault(s.speaker_id, []).append(s)
        import numpy as np
        multi = [k for k, v in by_spk.items() if len(v) >= 2]
        if len(multi) >= 2:
            def emb(s):
                e = be.embed(load_mono(paths[s.sample_id], 16000)[0])
                return e / np.linalg.norm(e)
            a1, a2 = by_spk[multi[0]][:2]
            b1 = by_spk[multi[1]][0]
            ea1, ea2, eb1 = emb(a1), emb(a2), emb(b1)
            print(f"embedding dim = {ea1.shape[0]}")
            print(f"same speaker ({multi[0]}): cosine = {float(ea1 @ ea2):.3f}")
            print(f"different speakers ({multi[0]} vs {multi[1]}): cosine = {float(ea1 @ eb1):.3f}")
        else:
            print("need two speakers with >=2 samples in the manifest for this check")


if __name__ == "__main__":
    main()
