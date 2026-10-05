# ViNV-TTS Evaluation Framework — System Description (v2.1)

You are working on an evaluation codebase for the **Vietnamese Non-Verbal Text-to-Speech for Conversational Synthesis (ViNV-TTS)** shared task (VLSP 2026).

This codebase is for **model development and internal evaluation**. It is NOT a reproduction of the organizers' official evaluator.

This v2.1 replaces v1; sections 1–20 are the v2 design and section 21 records what was built, measured and decided during implementation. It keeps v1's principles and incorporates (a) facts verified on the released data, (b) corrections to v1, and (c) decisions made since. Section 20 lists what is still undecided.

---

## 1. Scope

The evaluator evaluates **one model's outputs per run**.

```text
Input : evaluation manifest + one model's generated audio (+ reference info)
Output: metrics, diagnostics, scores, report (+ optional human-evaluation data)
```

The evaluator MUST NOT: compare models, rank/leaderboard, track experiments or keep a database, compute improvements between runs, or select models. Comparison happens outside this codebase.

Platform note: the developer works on Windows (`D:\Work\...`). Use `pathlib`, open all text files with `encoding="utf-8"`, and avoid shell-specific assumptions.

---

## 2. Task Background

Input: Vietnamese text with inline NV tags (`[laughter]`, `[breathing]`, `[sniff]`, `[throatclearing]`). Output: conversational speech where each tagged NV is present, of the correct type, at the intended position, natural, intelligible, and in the target speaker's voice.

- **Track A (Core):** speaker seen during training.
- **Track B (Advanced):** unseen speaker given by a short reference clip (zero-shot).

**One run evaluates exactly one track.** The track is a run-level setting (`track: A | B`) validated against the manifest. Reports never show Track A and Track B scores side by side.

---

## 3. Dataset Facts (verified from the released train/dev data)

Layout:

```text
<root>/<split>/<spk_id>/<spk_id>.json     # list of {"audio", "text", "language_id"}
<root>/<split>/<spk_id>/<audio files>.flac
split ∈ {train, dev}
```

- Metadata has ONLY `audio`, `text`, `language_id` (always `vi`). **No speaker field** (speaker = folder name), **no duration, no timestamps, no NV timing, no reference-clip field.**
- The same `spk_id` in train and dev is the same person.

| | train | dev |
|---|---|---|
| speakers | 189 | 46 (all also in train) |
| utterances | 1739 | 316 |
| hours | 5.90 | 1.04 |
| NV events | 4646 | 809 |
| utterances without NV | 0 | 0 |
| dev `clean_text` found in train | — | 0 / 316 |

- NV counts (train / dev): breathing 3733 / 674, laughter 681 / 102, sniff 117 / 14, throatclearing 115 / 19.
- Audio: all 24 kHz mono FLAC; mix of PCM_24 (~75%) and PCM_16. Duration 2.6–25.8 s, median ~12.7 s. Median ~50 words/utterance (words = whitespace tokens, i.e. Vietnamese syllables), ~4.07 words/s. Dev matches train distributionally.
- NVs per utterance: 1–10 (train). NV position (train): start 2.2%, middle ~95%, end 3.1%. Median distance between consecutive NVs: 12 words; 85 train pairs of NVs sit in the *same gap* between two words (51 of them sniff+breathing); ~5.5% of consecutive pairs are ≤1 word apart.
- **Speaker distribution is extremely skewed.** Train: 6 speakers ≈ 76% of utterances (`spk_0000` alone 32%; 37% of dev); 101 speakers have a single utterance; 173 speakers have ≤5 utterances (0.95 h total). Dev: 30 of 46 speakers have one utterance. Tail speakers have only ~3–78 s of audio.
- Text format (clean): NV tags are always whitespace-delimited (never glued to words or punctuation), no unbalanced brackets, no digits, NFC, almost entirely lowercase (some capitalised sentence starts), punctuation present in most utterances.
- **Transcripts and tags are produced by an automated pipeline** (per the task description; consistent with observed ASR-like errors on proper nouns). Therefore even ground-truth audio will NOT achieve WER 0 or NVPA 1.

Anything not listed above has not been verified; do not assume it.

---

## 4. Official vs Development Assumption

Every parameter and behaviour in code and config must be labelled one of:

```text
[OFFICIAL]    stated in the task description
[ASSUMPTION]  our development choice; configurable; recorded in every report
```

Reports must print the active assumptions. Never claim the internal NVPA (or any score) matches the organizers' implementation.

---

## 5. Metrics

| Metric | Source | Range | Direction | Notes |
|---|---|---|---|---|
| NVPA | auto (dev approximation) | 0–1 | higher | required NV type present at intended position |
| SN | human | 1–5 | higher | includes speech around NVs |
| Q | human | 1–5 | higher | artifacts, noise, distortion |
| WER | auto, Zipformer `[OFFICIAL]` | 0–1 | lower | NV tags removed from reference |
| pMOS | auto, DNSMOS `[OFFICIAL]` | 1–5 | higher | |
| SS | auto, ECAPA-TDNN `[OFFICIAL]` | 0–1 | higher | |

---

## 6. Scoring

All components are normalized to [0,1].

```text
Track A: A = 0.30 NVPA + 0.15 SN + 0.15 Q + 0.15 (1-WER) + 0.15 pMOS + 0.10 SS
Track B: B = 0.30 NVPA + 0.15 SN + 0.15 Q + 0.10 (1-WER) + 0.10 pMOS + 0.20 SS
```

Normalization rules:

- `[ASSUMPTION]` SN, Q, pMOS: `(x - 1) / 4`. The organizers say components are normalized to [0,1] but do not give the formula.
- `[ASSUMPTION]` `1 - WER` is computed on WER clipped to [0,1] (raw WER can exceed 1; keep the raw value in reports).
- `[ASSUMPTION]` SS: cosine similarity clipped to [0,1] (raw value kept in reports).

### Automatic Score (Fast mode)

Only the available components, **same weights as the official formula, no renormalization in the headline value**:

```text
AutoScore_A = 0.30 NVPA + 0.15 (1-WER) + 0.15 pMOS_norm + 0.10 SS
AutoScore_B = 0.30 NVPA + 0.10 (1-WER) + 0.10 pMOS_norm + 0.20 SS
```

Important: the weights sum to **0.70**, so AutoScore has a maximum of 0.70, not 1. Also report `AutoScore / 0.70` clearly labelled "renormalized (convenience only)". AutoScore must never be labelled official or final.

### Final Score (Full mode)

Computed only when both SN and Q are available. Never substitute pMOS for SN/Q, never duplicate pMOS, never fabricate. If unavailable: `Final Score: N/A`.

When SN/Q come from a human-evaluation *subset*, the automatic components in the final score must be computed **on that same subset** (default `[ASSUMPTION]`; configurable). Report both subset-based and full-set automatic metrics.

### Test vectors (recompute by hand in unit tests; do NOT copy numbers from older documents)

Track A, NVPA 0.713, WER 0.043, pMOS raw 4.12, SS 0.871:

```text
1-WER = 0.957 ; pMOS_norm = (4.12-1)/4 = 0.780
AutoScore_A = 0.3*0.713 + 0.15*0.957 + 0.15*0.780 + 0.10*0.871 = 0.56155
AutoScore_A / 0.70 = 0.802

with SN 4.10 -> 0.775 ; Q 4.25 -> 0.8125:
A = 0.56155 + 0.15*0.775 + 0.15*0.8125 = 0.799675
```

---

## 7. Evaluation Modes and Caching

**Fast mode:** NVPA, WER, pMOS, SS → `AutoScore`. Does not produce SN, Q, or a final score. Report states: "Human evaluation: not available. Final score: N/A".

**Full mode:** Fast results + human SN/Q (+ optional diagnostic ratings) → final score.

Human evaluation is an additional stage and must NOT re-run automatic evaluation. Each metric writes its own artifact:

```text
artifacts/<metric>/per_sample.jsonl   + meta (hash of generated audio set, manifest, metric config)
```

The `score` stage only reads artifacts. A metric is recomputed only if its input hash changes.

---

## 8. Data Layer

### 8.1 Adapter → canonical manifest

Layout-specific code lives only in a **dataset adapter** that converts the released data (and model outputs) into the canonical manifest. The rest of the evaluator knows only the manifest. Fields:

```text
sample_id, track, speaker_id,
text                 # raw, with tags
clean_text           # tags removed (produced by the canonical parser)
nv_events            # list of {type, gap_index}
reference_audio      # path(s); see 8.3
generated_audio      # path
(optional) ground_truth_audio, duration
```

### 8.2 One canonical NV parser

All components consume parser output; no other module parses raw tags.

- Tags are matched as `[...]`; the four known types are configurable. Unknown/malformed tags are reported and handled by a configurable policy (`error | drop | keep_as_unknown`); data currently contains none.
- `clean_text`: tags removed, whitespace collapsed. Words = whitespace tokens.
- `gap_index` = number of words before the tag (0 = before the first word, `n_words` = after the last). Several NVs may share one gap; their order is preserved in the list but treated per matcher config (see 9.4).
- Handles: tag at start/end, multiple tags, adjacent tags, same-gap tags of different types.

### 8.3 Reference audio

The released metadata has no reference-clip field.

- Track B: the reference clip is supplied with the evaluation input.
- Track A: `[ASSUMPTION]` reference = speaker centroid embedding from that speaker's *training* audio (rule configurable: centroid / single clip / N clips). Report how many seconds of reference audio each speaker has; tail speakers have as little as ~3 s.

---

## 9. NVPA (development approximation)

### 9.1 Pipeline

```text
generated audio
   ├─ ASR (Zipformer) → recognized words + timing   (shared with WER)
   └─ NV detector     → [(type, start, end, confidence)]
gold: clean_text words + nv_events (type, gap_index)
   → map detected NV times to gap indices in the recognized transcript
   → map recognized transcript to clean_text via edit-distance alignment
   → match detected vs gold events
```

Do not assume the Zipformer checkpoint exports word-level timing. If it exports only token-level timing, derive word timing from tokens; if timing is unavailable, fall back to forced alignment. Source of timing is configurable. Define behaviour when ASR is poor (low alignment confidence): mark the sample as `alignment_unreliable` and report it, do not silently score.

### 9.2 Detector (pluggable, none assumed adequate)

Define a detector interface. Candidates (e.g. pretrained audio-event tagger; classifier trained on the corpus with weak labels derived from ASR timing) are only trusted after the calibration runs in section 13. Detector choice is open (section 20).

### 9.3 Configuration

```text
nvpa:
  detector:            <plugin + checkpoint>
  timing_source:       <asr_tokens | forced_alignment | ...>
  matching:            <e.g. greedy | optimal assignment>
  position_tolerance:  <in words and/or seconds>
  type_matching:       exact
  same_gap_order:      unordered   # [ASSUMPTION]
  aggregation:         micro       # headline NVPA [ASSUMPTION]
```

### 9.4 Matching notes (from the data)

- One word ≈ 0.25 s; consecutive NVs are typically ~12 words apart, so a loose tolerance can match by chance. See the shuffle baseline (section 13).
- Same-gap NVs of different types (e.g. sniff + breathing) are common enough to need explicit handling; ~5.5% of consecutive NV pairs are ≤1 word apart, so the detector must separate nearby events.

### 9.5 Per-event record and failure taxonomy

For every gold NV event store: gold type, gold gap index, matched detection (if any), position error (words and seconds), detected duration, and a failure reason:

```text
ok | missing | wrong_type | wrong_position | alignment_unreliable
```

Spurious (extra, unrequested) NVs are reported separately and do NOT enter NVPA.

### 9.6 Aggregation and uncertainty

- Headline NVPA: micro average over events `[ASSUMPTION]`. Always also report macro over NV types and macro over speakers.
- Breathing is ~80% of events, so micro NVPA mostly reflects breathing; the per-type figures are essential.
- Dev has only 14 sniff and 19 throatclearing events (one event ≈ 5–7 points). Every NVPA figure is reported with its `n` and a bootstrap confidence interval.

---

## 10. WER

```text
generated audio → Zipformer → ASR text → normalization → WER vs normalized clean_text
```

- NV tags are removed from the reference (via the canonical parser) `[OFFICIAL]`.
- Normalization `[ASSUMPTION]`, configurable: Unicode NFC, lowercase, strip punctuation, collapse whitespace. Data contains no digits.
- Tokenization: whitespace tokens (syllables).
- Zipformer checkpoint/config is configurable. Store the ASR transcript per sample.
- Report raw WER (may exceed 1) and the clipped value used in scoring.
- Reference text is itself automatically produced and noisy, so a non-zero WER floor is expected (see calibration).

---

## 11. pMOS

DNSMOS or a selected MOS model, configurable. Resample to the model's required rate (24 kHz → 16 kHz for DNSMOS). Verify how the chosen implementation handles audio longer than its analysis window (utterances reach ~26 s). The output used (e.g. OVRL vs SIG/BAK) is configurable `[ASSUMPTION]`. Report raw and normalized pMOS. Never use as SN/Q.

---

## 12. Speaker Similarity

```text
reference audio → ECAPA-TDNN → reference embedding
generated audio → ECAPA-TDNN → generated embedding → cosine similarity
```

Checkpoint, preprocessing, sample rate, VAD, and similarity method are configurable. Report per-sample values and mean, median, std, min, max, percentiles. Report per speaker (and the amount of reference audio per speaker). Do not hide per-sample values.

---

## 13. Calibration Runs (new)

A calibration run is an ordinary single-model run with a different audio source; the tool does not compare it with anything. Researchers interpret model results against it externally.

1. **Ground-truth run:** treat the dev ground-truth audio as generated audio. Yields the practical ceiling for WER, NVPA, pMOS, SS, given noisy transcripts and tags. Also validates the NV detector and the position mapping.
2. **NVPA shuffle baseline:** keep the audio, shuffle gold NV positions (within the same utterance), recompute NVPA. Shows how much NVPA a given tolerance gives by chance; use it to tune tolerance.
3. **Optional degradation checks:** e.g. audio with NVs removed or replaced, to confirm NVPA drops.

Do not trust any NVPA configuration that has not passed 1–2.

---

## 14. Diagnostics (modular, not mixed with the official score)

Breakdowns, each with `n` and a flag when `n` is below a configurable threshold:

```text
overall · by NV type · by speaker (micro and macro) · speaker group (head vs tail)
by utterance length · by number of NVs · by NV position (start / middle / end)
```

Position classes start/end are sparse in dev (13 and 27 events); treat them as indicative only. NV Detection Rate, NV Type Accuracy, NV Placement Error, NV Duration are exposed per type.

Not in scope (decided): "with NV vs without NV" comparison and any control synthesis; every data utterance contains NVs, and Zipformer is expected to ignore NVs. The calibration ground-truth run will show whether ASR transcripts near NVs look unusual.

Do not over-engineer: the core metrics come first.

---

## 15. Human Evaluation

Fixed subset selected with a seed and saved. Stratify over: NV type (include rare sniff/throatclearing utterances), NV position, head vs tail speakers, utterance length, NV count.

Absolute ratings per sample (1–5): **SN**, **Q** (required); **NV Naturalness**, **NV Placement** (diagnostic; not in the official formula unless explicitly configured). Optional pairwise comparison, never affecting the official score unless configured.

- Randomized, anonymized presentation; annotators never see model identity.
- **Rater sanity anchors:** insert hidden ground-truth items (and optionally degraded items); flag raters whose anchor ratings are implausible.
- Store per-rater scores; report rater count and spread. Aggregate: mean per sample, then mean over samples.

Data flow: `auto artifacts → human subset → human_scores.json → combine → final score`. The annotator interface is decided later (open).

---

## 16. Output Structure

```text
evaluation_output/
  summary.json
  per_sample.csv|json
  artifacts/<metric>/...
  diagnostics/
  human_evaluation/
  report/
```

`summary.json` keeps these blocks separate: `automatic_metrics`, `human_metrics`, `automatic_score`, `final_score`, `active_assumptions`, `data_counts`.

Example report (Track A):

```text
Automatic Metrics                       (n=316, 95% CI in summary.json)
  NVPA 0.713 | WER 0.043 (1-WER 0.957) | pMOS 4.12 (norm 0.780) | SS 0.871
Automatic Score  0.562   (max 0.70; renormalized 0.802)

Human Metrics:   SN N/A | Q N/A
Final Score:     N/A
```

---

## 17. Engineering Principles

Reproducibility; transparent assumptions; modular evaluators; per-sample results; configurable checkpoints (never hard-coded); deterministic evaluation (fixed seeds, fixed subset); lazy-load heavy models; a failure on one sample is recorded and does not abort the run; clear separation of automatic and human evaluation.

Proposed layout:

```text
nvtts_eval/
  data/      adapter, nv_parser, manifest
  metrics/   wer, pmos, speaker_sim, nvpa/{detector, mapping, matcher}
  scoring/   auto_score, final_score         (pure functions)
  human/     subset, export, import, anchors
  report/
  cli, configs/
tests/       parser edge cases, scoring test vectors (section 6), matcher cases
scripts/     dataset_stats.py (already written)
```

The evaluator should tell the researcher not only how well the model performs but **where and why it fails**.

---

## 18. Constraints — Do NOT

1. Invent official evaluator details that were not provided.
2. Claim the internal NVPA matches the organizers'.
3. Use pMOS as SN or Q.
4. Produce a final score without human SN/Q.
5. Add extra metrics to the official formula without explicit configuration.
6. Build model comparison, ranking, or experiment tracking.
7. Assume dataset properties not listed in section 3.
8. Hard-code model checkpoints.
9. Mix research diagnostics into the official score.
10. Label AutoScore as official, or present it on a [0,1] scale without stating its 0.70 ceiling.

---

## 19. Development Phases

0. **Dataset understanding — done** (`dataset_stats.py`; results in section 3).
1. **Data layer:** canonical NV parser, adapter, manifest, config skeleton, artifact/caching framework, scoring functions + test vectors.
2. **Easy automatic metrics:** WER, pMOS, SS; AutoScore; ground-truth calibration run for these.
3. **NVPA:** timing/mapping, detector plugin(s), matcher, per-event records, shuffle baseline, validation on ground-truth audio.
4. **Diagnostics:** breakdowns with `n` and confidence intervals.
5. **Human evaluation:** subset, anonymization, anchors, import/export.
6. **Final scoring and reporting.**

---

## 20. Open Decisions

- NVPA detector choice and default tolerance (to be set from calibration results).
- Source of word/token timing from the chosen Zipformer checkpoint.
- Whether the organizers' NVPA is micro or macro (unknown; ours is a configurable assumption).
- DNSMOS variant/output and handling of long audio.
- Track A reference-audio rule for speakers with very little training audio.
- Local protocol for Track B evaluation: **deferred** (the released dev set has no unseen speakers; all 46 dev speakers are in train).
- Human-evaluation interface and subset size.
- Exact normalization the organizers apply to SN, Q, pMOS, SS, WER.

---

## 21. As Built (v2.1)

### 21.1 Checkpoints (all verified end to end on Kaggle, T4, Python 3.13)

- **ASR:** `hynt/Zipformer-30M-RNNT-6000h`, offline transducer, ONNX fp32, run with `sherpa-onnx`. The repo ships no
  `tokens.txt`; it is generated from `bpe.model` (2000 tokens: `<blk> 0`, `<sos/eos> 1`, `<unk> 2`, ...). Output tokens are
  upper case (normalisation lower-cases). **Token timestamps are available** (0.04 s grid, onset of each token). Decoding
  takes about 0.2–0.6 s for 5–14 s clips on CPU. Weights are CC BY-NC-ND 4.0: local use only.
- **pMOS:** `prj-beatrice/dnsmos-torch-native` (remote code, pin `revision`): outputs `sig`, `bak`, `ovrl`, `p808`; a 24.7 s
  clip ran without error. `ovrl` is the default pMOS `[ASSUMPTION]`.
- **Speaker similarity:** `speechbrain/spkrec-ecapa-voxceleb`: 192-d embeddings; probe: same speaker 0.748, different
  speakers 0.088. Device string must be `cuda:0` (fixed).
- All models receive mono float32 at 16 kHz from one resampling function (`audio.load_mono`).

### 21.2 Ground-truth calibration of the automatic metrics (dev audio as "model output", Track A)

316 utterances, 46 speakers; reference for SS = first 20 train clips of the speaker (centroid).

| metric | value | 95% CI (utterance) | 95% CI (speaker-cluster) |
|---|---|---|---|
| WER (corpus: 414 errors / 15,091 words; S 238, D 133, I 43) | 0.027 | [0.024, 0.031] | [0.022, 0.042] |
| pMOS, DNSMOS `ovrl` | 3.13 (normalised 0.533) | [3.095, 3.160] | [3.026, 3.186] |
| SS, cosine | 0.810 (macro-speaker 0.772) | [0.799, 0.820] | [0.758, 0.832] |

Consequences: real speech reaches only 0.533 on normalised pMOS and 0.81 on SS, so these are the practical ceilings.
Automatic components without NVPA contribute at most 0.15·(1−0.027) + 0.15·0.533 + 0.10·0.810 = **0.307** of 0.40
(Track A); with NVPA at its own ground-truth level `x`, the ground-truth AutoScore is 0.307 + 0.30·x. The low WER suggests the
corpus transcripts were produced by an ASR pipeline close to this one, so WER on real speech is a lower bound that a TTS
system should not be expected to match.

### 21.3 NVPA as built

1. ASR (shared with WER) gives words with token timestamps.
2. Reference and recognised words are aligned (edit alignment). Each gap between two reference words maps to the window
   `[onset of the last token of the previous word, onset of the first token of the next word]`
   (at least 0.2 s, at most 3 s; widened and flagged when a neighbouring word was not recognised; none if nothing aligned).
3. A window-level detector (one gradient-boosting classifier per NV type on 38 hand-crafted acoustic features,
   `FEATURE_VERSION 1`) gives per-window presence probabilities. It is trained **only on the train split** (windows from the
   ASR alignment of ground-truth audio), thresholds are chosen on speaker-grouped out-of-fold F1; dev is never used for
   training. The detector is replaceable (`predict_windows`, `thresholds`, `nv_types`, `describe`).
4. Gold events `(type, gap)` are matched one-to-one to detections of the same type within `tolerance_words` (default 1
   `[ASSUMPTION]`), closest first. Misses are classified: `alignment_unreliable`, `wrong_type`, `wrong_position`, `missing`.
   Headline NVPA = micro average over events `[ASSUMPTION]`; events that cannot be localised count as misses
   (`unreliable_policy: miss`, alternative `exclude`); failed samples are excluded and counted.
5. Reported with it: per type, per position class, macro over types and speakers, detection rate, type accuracy given
   detection, placement error (words, seconds), spurious NVs per 100 words (not part of the score), bootstrap CIs
   (utterance and speaker-cluster) and the **random-placement baseline** (gold types kept, positions drawn uniformly, same
   detections), which must be read next to NVPA. `nvpa-sweep` re-evaluates a stored artifact under other tolerances and
   thresholds without running any model, and prints the signed offset between each gold event and the nearest detection of
   its type.
6. Checked on synthetic audio (crude artificial NVs in the gaps): NVPA 0.91 against a random-placement baseline of 0.29 and
   no spurious detections. This verifies the chain, not the detector's quality on real recordings.

Limitations: presence per window only (NV *duration* from spec section 10 is not measured); two same-type events in one gap
cannot both be matched; timing is the onset grid of the ASR (0.04 s); tags in the corpus are themselves automatic, so the
detector learns the pipeline's notion of an NV (circularity) and the ground-truth NVPA is below 1.

### 21.4 NVPA calibration on the real corpus (dev ground truth as "model output", Track A, tolerance 1 word)

Detector, speaker-grouped out-of-fold on train (88,363 windows from 1,739 utterances; positives: breathing 3,722,
laughter 681, sniff 117, throat clearing 115):

| type | threshold | precision | recall | AUC |
|---|---|---|---|---|
| breathing | 0.888 | 0.71 | 0.72 | 0.957 |
| laughter | 0.683 | 0.31 | 0.27 | 0.851 |
| sniff | 0.542 | 0.22 | 0.42 | 0.951 |
| throat clearing | 0.498 | 0.14 | 0.15 | 0.907 |

NVPA on dev (316 utterances, 809 gold events):

| quantity | value |
|---|---|
| NVPA, micro | **0.729** (95% CI utterance [0.695, 0.762]; speaker-cluster [0.604, 0.775]) |
| random-placement baseline (same detections) | 0.130 (std 0.014): lift 0.599 |
| per type | breathing 0.804 (n=674), laughter 0.402 (102), throat clearing 0.263 (19), sniff 0.143 (14) |
| macro over types / over speakers | 0.403 / 0.573 |
| detection rate; type accuracy given detection | 0.764; 0.955 |
| misses | wrong_position 120, missing 70, wrong_type 28, alignment_unreliable 1 |
| spurious NVs per 100 words | 1.75 |
| AutoScore of ground truth (Track A) | 0.525 of 0.70 (renormalised 0.751) |

Reading: the chain separates real placement from chance (about 5.6 times the baseline), but (a) breathing is 83% of the events
and 92% of the hits, so micro NVPA is mostly a breathing score; laughter, sniff and throat clearing are detected poorly
and sniff / throat clearing have only 14 / 19 dev events, so per-type values must always be read with their `n`;
(b) **55% of all misses are `wrong_position`**: a detection of the right type exists but lies farther than the tolerance,
which points to the tolerance, a systematic offset between corpus tags and the window definition, or both; `nvpa-sweep`
shows which. Because tolerance and threshold changes raise NVPA and the baseline together, they are chosen by the lift.
The detector was trained on natural speech; whether it transfers to the NV sounds of TTS systems is validated only by
comparing NVPA with the raters' `NV_placement` scores (the summary reports their Spearman correlation per sample).

### 21.5 Human evaluation as built

`human-subset` (deterministic; rare NV types first, then strata head/tail speaker × short/long with sqrt-proportional quotas
and round-robin over speakers) → `human-export` (anonymous audio names, `rating_sheet.csv`, Vietnamese instructions,
hidden ground-truth and noise-degraded anchors; `PRIVATE_key.json` stays with the organiser) → `human-import` (one sheet per
rater; rater id = file name; range checks; anchor check flags a rater whose ground-truth anchors are rated below
`anchor_gt_min` or less than `anchor_gap_min` above the degraded ones; inter-rater agreement; per-sample means over accepted
raters). The final score uses the automatic components computed on the rated subset (`auto_on_subset`); full-set values
are reported separately.

### 21.6 Decisions

Resolved: ASR timing source (token timestamps); DNSMOS output (`ovrl`, configurable); Track A reference rule (centroid of
train clips, optionally capped); human-evaluation interface (CSV package); detector family (window-level gradient boosting,
subject to the calibration below).

Still open: choice of tolerance (and optional global threshold scale) from `nvpa-sweep`; detector quality for laughter, sniff
and throat clearing (if it stays weak, replace the feature extractor with a pretrained audio encoder behind the same interface);
transfer of the detector to TTS output (human `NV_placement` correlation); whether the organizers' NVPA is micro or macro; how they normalise SN, Q, pMOS,
SS; Track B local protocol; adapters for the public/private test formats once released.

