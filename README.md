# nvtts-eval

Single-model evaluator for the ViNV-TTS shared task (VLSP 2026). Design and as-built notes: `NVTTS-Eval-Spec-v2.md`.

All components exist: parser/manifest/adapter, WER, pMOS, speaker similarity, NVPA, human-evaluation package,
final scoring, CLI. Tests: `python -m pytest -q` (154). The three model wrappers, the detector and the full calibration run were verified
on Kaggle; results are in section 21 of the spec (ground-truth NVPA 0.729 vs a random-placement baseline of 0.130).

## Checkpoints (configurable in `configs/default.yaml`)
| role | checkpoint | runtime |
|---|---|---|
| ASR (WER, token timing for NVPA) | `hynt/Zipformer-30M-RNNT-6000h` (ONNX) | `sherpa-onnx` |
| pMOS | `prj-beatrice/dnsmos-torch-native` | `torch` + `transformers` (remote code: pin `revision`) |
| Speaker similarity | `speechbrain/spkrec-ecapa-voxceleb` | `torch` + `speechbrain` |

ASR weights are CC BY-NC-ND 4.0: use locally, never commit them. All models get mono float32 at 16 kHz.

## Setup
    pip install -e ".[dev,asr,mos,asv,nvpa]"        # install torch first for your CUDA/CPU setup
    python -m pytest -q
    python scripts/make_tokens.py --bpe <asr dir>/bpe.model --out <asr dir>/tokens.txt     # once; the HF repo has no tokens.txt
    python scripts/probe_models.py --manifest <manifest> --config <config> --n 3           # checks the three models

## 1. Calibration on the real corpus (do this once; ground-truth audio = "model output")
    # manifests (dev for evaluation, train for training the detector)
    python -m nvtts_eval.data.adapter_vinv --root <data> --split dev   --track A --out manifests/dev_gt_A.jsonl
    python -m nvtts_eval.data.adapter_vinv --root <data> --split train --track A --no-references --out manifests/train_gt.jsonl

    # train the NVPA detector on TRAIN only (runs the ASR over train once; cached in the run dir)
    python -m nvtts_eval.cli train-detector --manifest manifests/train_gt.jsonl --run-dir runs/train_gt --config <config> --out models/nvpa_gap.joblib

    # set nvpa.detector_path in the config, then run everything on dev (asr/pmos/ss are reused from cache)
    python -m nvtts_eval.cli run --manifest manifests/dev_gt_A.jsonl --run-dir runs/gt_dev_A --config <config>

Then diagnose the tolerance and thresholds without running any model:

    python -m nvtts_eval.cli nvpa-sweep --run-dir runs/gt_dev_A --config <config>

Read the result before trusting NVPA: the detector's out-of-fold precision/recall per NV type (printed by
`train-detector`), the ground-truth NVPA (the practical ceiling), the random-placement baseline printed beneath it, and the
spurious-NV rate. A detector whose ground-truth NVPA is close to the random-placement baseline cannot rank models.

## 2. Evaluate a model
Put one audio file per sample in a folder, named `<spk_id>_<audio stem>.wav` (e.g. `spk_0000_0001.wav`), then:

    python -m nvtts_eval.data.adapter_vinv --root <data> --split dev --track A --source model --generated-dir <outputs> --out manifests/mymodel_A.jsonl
    python -m nvtts_eval.cli run --manifest manifests/mymodel_A.jsonl --run-dir runs/mymodel_A --config <config>

Track B needs `--track B --reference-map refs.json` (`{sample_id: [reference clip paths relative to --root]}`): the released
data has no unseen speakers. Missing or failed samples are listed as errors and counted, never silently dropped.
Without human scores the report shows AutoScore (maximum 0.70, not official) and `Final score: N/A`.

## 3. Human evaluation (adds SN, Q and the final score)
    python -m nvtts_eval.cli human-subset --manifest <m> --run-dir <run> [--size 100]
    python -m nvtts_eval.cli human-export --manifest <m> --run-dir <run> --out <package dir>
    # give raters ONLY <package dir>/for_raters ; each rater fills a copy of rating_sheet.csv and returns it as <rater>.csv
    python -m nvtts_eval.cli human-import --manifest <m> --run-dir <run> --package <package dir> --ratings alice.csv bob.csv

Keep `PRIVATE_key.json` away from raters. Raters whose hidden anchors look wrong are flagged and excluded (config).
If raters fill the optional `NV_placement` column, the summary also prints the per-sample Spearman correlation between NVPA and
that score: the most direct check that the NVPA detector is trustworthy on TTS output.

## Other
    python -m nvtts_eval.cli summary --manifest <m> --run-dir <run>        # rebuild the report from cached artifacts
    python scripts/dataset_stats.py --root <data> --out stats_out
Use `--limit N` with a separate `--run-dir` for smoke tests. `--force` recomputes a metric.
