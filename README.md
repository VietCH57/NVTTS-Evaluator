# nvtts-eval

Single-model evaluator for the ViNV-TTS shared task. Design: `NVTTS-Eval-Spec-v2.md`.

## Status
Phase 1 done (parser, manifest, adapter, scoring, artifact cache, statistics).
Phase 2 in progress: ASR+WER, pMOS, speaker similarity, summary, CLI are written and tested with
fake backends; **the real-model wrappers still need to be verified on your machine** with
`scripts/probe_models.py` (see below).
Not implemented yet: NVPA (so AutoScore is reported as N/A), human evaluation.

## Checkpoints (configurable in `configs/default.yaml`)
| role | checkpoint | runtime |
|---|---|---|
| ASR (WER, token timing for NVPA) | `hynt/Zipformer-30M-RNNT-6000h` (offline transducer, ONNX) | `sherpa-onnx` |
| pMOS | `prj-beatrice/dnsmos-torch-native` | `torch` + `transformers` (remote code: pin `revision`) |
| Speaker similarity | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN) | `torch` + `speechbrain` |

The ASR weights are CC BY-NC-ND 4.0: use them locally, do not commit or redistribute them.
All models receive mono float32 audio resampled to 16 kHz by `nvtts_eval.audio.load_mono`.

## Setup (PowerShell, from this folder)
    pip install torch                       # or the CUDA build from pytorch.org
    pip install -e ".[dev,asr,mos,asv]"
    python -m pytest -q

Download the ASR files (`encoder-epoch-20-avg-10.onnx`, `decoder-...onnx`, `joiner-...onnx`, `bpe.model`;
or the `.int8.onnx` variants) into one folder, then create the missing `tokens.txt`:

    python scripts\make_tokens.py --bpe D:\Work\models\Zipformer-30M-RNNT-6000h\bpe.model --out D:\Work\models\Zipformer-30M-RNNT-6000h\tokens.txt

Edit `asr.model_dir` (and `ss.savedir`) in `configs/default.yaml`.

## Verify the real models (do this first)
    python scripts\probe_models.py --manifest manifests\dev_gt_A.jsonl --config configs\default.yaml --n 3

It checks: ASR transcripts vs references and whether token timestamps exist (NVPA needs them);
DNSMOS outputs incl. on the longest clip; ECAPA cosine same-speaker vs different-speaker.

## Ground-truth calibration run (dev audio treated as the model output, Track A)
    python -m nvtts_eval.cli run --manifest manifests\dev_gt_A.jsonl --run-dir runs\gt_dev_A --config configs\default.yaml
    python -m nvtts_eval.cli summary --manifest manifests\dev_gt_A.jsonl --run-dir runs\gt_dev_A --config configs\default.yaml

Use `--limit 20` (with its own `--run-dir`) for a quick smoke test. Results are cached per metric;
unchanged inputs are never recomputed; `--force` recomputes.

## Other commands
    python -m nvtts_eval.data.adapter_vinv --root <data root> --split dev --track A --out manifests\dev_gt_A.jsonl
    python scripts\dataset_stats.py --root <data root> --out stats_out
