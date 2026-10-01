# ViNV-TTS Evaluation Framework — Project Context & Requirements

You are working on an evaluation codebase for the **Vietnamese Non-Verbal Text-to-Speech for Conversational Synthesis (ViNV-TTS)** shared task.

The purpose of this codebase is **model development and internal evaluation**, not exact reproduction of the organizer's official evaluator.

The evaluator should allow researchers to run evaluation on **one model/output at a time** and obtain a structured set of metrics. Comparing multiple models or experiments is outside the scope of this codebase.

---

## 1. Task Background

The system receives Vietnamese text containing inline non-verbal vocalization (NV) tags such as:

* `[laughter]`
* `[breathing]`
* `[sniff]`
* `[throatclearing]`

and synthesizes conversational Vietnamese speech.

The system should:

* preserve the linguistic content;
* realize the requested NV type;
* place the NV at the intended position;
* produce natural and high-quality speech;
* preserve the target speaker identity.

There are two tracks:

### Track A — Core

The target speaker is seen during training.

### Track B — Advanced

The target speaker is unseen and is specified using a short reference audio clip. Zero-shot voice cloning is required.

---

# 2. Organizer Metrics

The shared task defines the following metrics.

### NV Placement Accuracy (NVPA)

Measures whether required NV events are correctly realized at their intended positions.

Range:

```text
0–1
```

Higher is better.

An NV event is considered correctly realized when:

* the required NV type is present;
* it occurs at the intended position.

The exact official implementation is not necessarily available. Therefore, this codebase should implement a **development-oriented approximation**, and all assumptions must be explicit/configurable.

---

### Speech Naturalness (SN)

Human perceptual rating.

Range:

```text
1–5
```

Higher is better.

---

### Quality (Q)

Human perceptual rating of audio quality, including artifacts, distortion, noise, etc.

Range:

```text
1–5
```

Higher is better.

---

### Word Error Rate (WER)

Measures speech intelligibility/content preservation.

The organizer description specifies Zipformer as the ASR model.

NV tags are excluded from WER computation.

Range:

```text
0–1
```

Lower is better.

---

### Predicted MOS (pMOS)

Automatic perceptual quality/naturalness prediction using DNSMOS.

Range:

```text
1–5
```

Higher is better.

---

### Speaker Similarity (SS)

Measures similarity between target/reference speaker and synthesized speech using ECAPA-TDNN.

Range:

```text
0–1
```

Higher is better.

---

# 3. Organizer Scoring Formula

All components are normalized to `[0,1]`.

## Track A

```text
A =
    0.30 * NVPA
  + 0.15 * SN
  + 0.15 * Q
  + 0.15 * (1 - WER)
  + 0.15 * pMOS
  + 0.10 * SS
```

where SN, Q and pMOS are normalized from `[1,5]` to `[0,1]`.

For example:

```text
normalized = (score - 1) / 4
```

## Track B

```text
B =
    0.30 * NVPA
  + 0.15 * SN
  + 0.15 * Q
  + 0.10 * (1 - WER)
  + 0.10 * pMOS
  + 0.20 * SS
```

---

# 4. Two Evaluation Modes

The framework MUST support two clearly separated modes.

## Mode 1 — Fast Evaluation

This mode is intended for frequent model development.

It uses only automatically computable metrics:

```text
NVPA
WER
pMOS
Speaker Similarity
```

It must NOT fabricate SN or Q.

Therefore, Fast Evaluation does **not** produce the official final score.

Instead, it produces an `Automatic Score`.

### Automatic Track A Score

Use only the available weighted components:

```text
AutoScore_A =
    0.30 * NVPA
  + 0.15 * (1 - WER)
  + 0.15 * pMOS_normalized
  + 0.10 * SS
```

### Automatic Track B Score

```text
AutoScore_B =
    0.30 * NVPA
  + 0.10 * (1 - WER)
  + 0.10 * pMOS_normalized
  + 0.20 * SS
```

These scores are useful for development but MUST NOT be labelled as the official/final competition score.

The report should explicitly state that human metrics are unavailable.

Example:

```text
Automatic Evaluation

NVPA                  0.713
WER                   0.043
1-WER                 0.957
pMOS                  4.12
pMOS normalized       0.780
Speaker Similarity    0.871

Automatic Score
Track A               0.555
Track B               0.514

Human Evaluation      Not available
Final Score            N/A
```

---

## Mode 2 — Full / Human Evaluation

Human evaluation provides:

```text
SN
Q
```

and optionally additional diagnostic human ratings such as:

```text
NV Naturalness
NV Placement
```

The official final score can only be computed once SN and Q are available.

The full score must use the organizer formulas above.

Do NOT substitute pMOS for SN or Q.

Do NOT duplicate pMOS to fill missing human metrics.

---

# 5. Important Design Principle

The evaluator is a **single-model evaluation tool**.

Each execution evaluates one model/output.

The evaluator MUST NOT:

* compare Model A vs Model B;
* maintain an experiment database;
* rank models;
* calculate improvements between experiments;
* generate leaderboards;
* manage model selection.

Researchers will perform model comparison outside this codebase.

The evaluator's responsibility is simply:

```text
Input:
    one model's generated outputs
    evaluation manifest
    required reference information

Output:
    metrics
    diagnostics
    optional human-evaluation data
    scores
    evaluation report
```

---

# 6. Recommended Evaluation Flow

The conceptual pipeline is:

```text
Evaluation Dataset
        |
        v
Generated Audio
        |
        v
+----------------------+
| Automatic Evaluation |
+----------------------+
        |
        +--> NVPA
        |
        +--> WER
        |
        +--> pMOS
        |
        +--> Speaker Similarity
        |
        v
Automatic Metrics
        |
        v
Automatic Score


Optional:

Evaluation Subset
        |
        v
Human Evaluation Interface
        |
        +--> SN
        +--> Q
        +--> NV Naturalness
        +--> NV Placement
        |
        v
Human Scores
        |
        v
Combine with automatic metrics
        |
        v
Final Track A / Track B Score
```

---

# 7. Do Not Re-run Automatic Evaluation During Human Evaluation

Human evaluation should be an additional stage.

For example:

### Step 1

Run Fast Evaluation:

```text
auto_metrics.json
```

### Step 2

Select a fixed human-evaluation subset.

### Step 3

Human annotators provide:

```text
human_scores.json
```

### Step 4

Combine the existing automatic metrics with human scores to calculate the final score.

Do not unnecessarily run Zipformer/DNSMOS/ECAPA again if the generated audio and automatic results have not changed.

---

# 8. Evaluation Dataset

The evaluator should operate from a structured manifest.

Conceptually each sample should contain:

```text
sample_id
text
clean_text
nv_events
speaker_id
reference_audio
generated_audio
```

For Track A, reference audio may be used for speaker evaluation according to the dataset setup.

For Track B, the target/reference speaker audio is explicitly required for speaker similarity.

The exact dataset structure should be discovered from the actual released training/dev data rather than assumed.

---

# 9. NV Representation

The input transcript contains inline tags.

For example:

```text
Hôm nay [laughter] tôi rất vui.
```

The parser should transform this into a structured representation such as:

```text
clean_text:
    Hôm nay tôi rất vui.

nv_events:
    type: laughter
    position: between "nay" and "tôi"
```

Do not make the rest of the evaluator independently parse raw NV tags.

There should be one canonical parser/representation shared by the evaluation components.

---

# 10. NVPA Should Have Diagnostic Metrics

Although the primary metric is NVPA, the evaluator should expose additional diagnostic information.

For example:

```text
NV Detection Rate
NV Type Accuracy
NV Placement Error
NV Duration
```

Potential breakdown:

```text
NVPA
    laughter
    breathing
    sniff
    throatclearing
```

Also consider contextual breakdowns:

```text
NV at beginning of utterance
NV in middle of utterance
NV at end of utterance
multiple NVs in one utterance
```

These diagnostic metrics are for research/debugging and do not automatically replace the main NVPA metric.

---

# 11. NVPA Implementation Must Be Configurable

The organizer description does not specify all details required for exact NV alignment.

Therefore, do NOT hard-code assumptions such as:

* a specific temporal tolerance;
* a specific NV detector;
* a specific alignment algorithm;
* a specific matching strategy.

Instead, make such parameters configurable.

For example conceptually:

```text
NVPA configuration

detector:
    <configurable>

matching:
    <configurable>

position_tolerance:
    <configurable>

type_matching:
    exact
```

The code should clearly distinguish:

```text
officially specified behavior
```

from:

```text
development assumption
```

---

# 12. WER Evaluation

The WER pipeline should conceptually be:

```text
Generated Audio
      |
      v
Zipformer ASR
      |
      v
Recognized Text
      |
      v
Text normalization
      |
      v
WER
```

NV tags must be removed from the reference transcript before WER calculation.

For example:

```text
Reference:

Hôm nay [laughter] tôi rất vui.

becomes:

Hôm nay tôi rất vui.
```

The exact Zipformer checkpoint/configuration should be configurable rather than assumed.

The evaluator should retain the ASR transcript for error analysis.

---

# 13. pMOS

Use DNSMOS or the selected automatic MOS model.

Report both:

```text
raw pMOS:          4.12
normalized pMOS:   0.780
```

Do not use pMOS as a substitute for human SN/Q.

The exact DNSMOS implementation/model should be configurable.

---

# 14. Speaker Similarity

Conceptually:

```text
Reference Audio
      |
      v
ECAPA-TDNN
      |
      v
Reference Embedding
      |
      |
Generated Audio
      |
      v
ECAPA-TDNN
      |
      v
Generated Embedding
      |
      v
Similarity
```

The exact ECAPA checkpoint, preprocessing, sample rate, VAD and similarity method should be configurable.

Report per-sample scores as well as aggregate statistics.

Useful diagnostics include:

```text
mean
median
std
min
max
percentiles
```

Do not hide the per-sample values.

---

# 15. Human Evaluation

Human evaluation should use a **fixed evaluation subset** so that different evaluation runs can use the same samples when researchers manually compare results outside the codebase.

The subset should be representative of:

* different NV types;
* different positions;
* different speakers;
* different utterance lengths;
* different NV counts.

The evaluator should support anonymous/randomized presentation to annotators.

Annotators should not be shown the model identity.

---

# 16. Human Evaluation Metrics

Required:

### Speech Naturalness

Scale:

```text
1–5
```

### Audio Quality

Scale:

```text
1–5
```

Useful additional diagnostic ratings:

### NV Naturalness

```text
1–5
```

### NV Placement

```text
1–5
```

These additional metrics are diagnostic unless explicitly incorporated into the competition formula.

Do not silently modify the official scoring formula to include them.

---

# 17. Human Evaluation Design

Human evaluation should support both:

### Absolute evaluation

Annotator hears one generated sample and rates it.

### Optional pairwise evaluation

Two anonymized samples can be compared.

Pairwise evaluation is optional and should NOT affect the official score unless explicitly configured.

---

# 18. Output Structure

Each evaluation run should produce a self-contained result.

Conceptually:

```text
evaluation_output/
    summary.json
    per_sample.json/csv
    diagnostics/
    human_evaluation/
    report/
```

The exact file format can be decided during implementation.

The summary should clearly distinguish:

```text
automatic metrics
human metrics
automatic score
final score
```

Example:

```text
Automatic Metrics
-----------------
NVPA               0.713
WER                0.043
pMOS               4.12
SS                 0.871

Human Metrics
--------------
SN                 4.10
Q                  4.25

Final Score
-----------
Track A            0.799
Track B            0.772
```

If human metrics are unavailable:

```text
Human Metrics
--------------
SN                 N/A
Q                  N/A

Final Score
-----------
Track A            N/A
Track B            N/A
```

---

# 19. Research-Oriented Diagnostics

The evaluator should retain enough information to answer questions such as:

* Which NV type fails most often?
* Are NVs missing or incorrectly typed?
* Are NVs generated at the wrong position?
* Does adding NVs increase WER?
* Which NV type causes the most intelligibility degradation?
* Does NV generation affect speaker similarity?
* Does speaker similarity differ significantly across speakers?
* Are errors concentrated in long or short utterances?
* Does the model perform differently for one NV position versus another?

Potential breakdowns:

```text
overall
by NV type
by speaker
by utterance length
by number of NVs
by NV position
with NV vs without NV
```

Do not over-engineer these analyses before the core evaluation works. They should be modular diagnostics.

---

# 20. Important Constraints

Do NOT:

1. Invent official evaluator details that have not been provided.
2. Claim that the internal NVPA implementation exactly matches the organizer.
3. Use pMOS as SN or Q.
4. Produce a fake "Final Score" when human SN/Q are unavailable.
5. Add extra metrics to the official scoring formula without explicit configuration.
6. Build model-comparison functionality into the evaluator.
7. Build an experiment-tracking/database system.
8. Assume the dataset structure before inspecting the actual released data.
9. Hard-code model checkpoints when they should be configurable.
10. Mix research diagnostics with the official score.

---

# 21. Development Priority

Implement conceptually in this order:

### Phase 1 — Dataset understanding

First inspect the actual released dataset and determine:

* metadata format;
* transcript format;
* NV tag format;
* speaker information;
* audio format;
* reference audio availability;
* split structure;
* whether alignment/timestamp information exists.

Do not start by assuming these properties.

### Phase 2 — Automatic evaluation

Implement:

```text
NVPA
WER
pMOS
Speaker Similarity
```

and the corresponding automatic scores.

### Phase 3 — NV diagnostics

Implement:

```text
NV detection
NV type accuracy
NV placement error
per-NV-type statistics
```

### Phase 4 — Human evaluation

Implement:

```text
fixed subset
anonymous samples
SN
Q
optional NV Naturalness
optional NV Placement
```

### Phase 5 — Final scoring

Combine automatic and human metrics according to the Track A / Track B formulas.

---

# 22. General Engineering Principle

The evaluator is a **research instrument**, not merely a competition submission script.

Prioritize:

* reproducibility;
* transparent assumptions;
* modular evaluators;
* per-sample results;
* configurable model/checkpoint paths;
* deterministic evaluation where possible;
* useful diagnostic information;
* clear separation between automatic and human evaluation.

Most importantly:

> The evaluator should tell the researcher not only **how well the model performs**, but also **where and why it fails**.

Before implementing any major component, inspect the actual dataset and existing repository structure. If a required assumption is not supported by the available data or task specification, do not invent it; flag it as a configurable design decision or ask for clarification.
