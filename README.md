# nvtts-eval

Single-model evaluator for the ViNV-TTS shared task. Design: see `NVTTS-Eval-Spec-v2.md`.

## Status
Phase 1 done: canonical NV parser, manifest, dataset adapter, scoring functions,
artifact store + metric runner (caching, per-sample error records), shared statistics
(describe, bootstrap CI incl. cluster bootstrap, per-group breakdown).
Not implemented yet: CLI, WER, pMOS, SS, NVPA, human evaluation.

## Setup
    pip install -e ".[dev]"          # core is stdlib-only
    python -m pytest -q

## Build a manifest from the released data (ground-truth calibration, Track A)
    python -m nvtts_eval.data.adapter_vinv --root D:\Work\NVTTS-Evaluator\data\vinv-tts --split dev --track A --out manifests\dev_gt_A.jsonl

(Run from the folder that contains `pyproject.toml`, after `pip install -e ".[dev]"`.
One line as above; PowerShell's line continuation is a backtick, not `^`.)

For a model run: add `--source model --generated-dir <dir with spk_0000_0001.wav, ...>`.

## Dataset statistics
    python scripts\dataset_stats.py --root <data root> --out stats_out
