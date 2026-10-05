# ViNV-TTS Evaluation Framework — System Description (v2)

You are working on an evaluation codebase for the **Vietnamese Non-Verbal Text-to-Speech for Conversational Synthesis (ViNV-TTS)** shared task (VLSP 2026).

This codebase is for **model development and internal evaluation**. It is NOT a reproduction of the organizers' official evaluator.

This v2 replaces v1. It keeps v1's principles and incorporates (a) facts verified on the released data, (b) corrections to v1, and (c) decisions made since. Section 20 lists what is still undecided.

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
