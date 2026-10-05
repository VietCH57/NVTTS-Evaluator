# nvtts-eval

**English** · [Tiếng Việt](#tiếng-việt) · [Linux](#linux)

---

## English

Single-model evaluator for the **ViNV-TTS** shared task (VLSP 2026, Vietnamese conversational TTS with non-verbal vocalizations).
It scores one system's synthesized speech with WER (Zipformer ASR), predicted MOS (DNSMOS), speaker similarity (ECAPA) and NVPA
(a development approximation of NV Placement Accuracy), and combines them with human ratings (SN, Q) into the task's score formula.
It is **not** the organizers' evaluator; every choice that the task description leaves open is an `[ASSUMPTION]`, configurable and
printed in every report. Design and measurements: `NVTTS-Eval-Spec-v2.md`.

### Status
Calibration is **done**. The detector and the operating point are stored in a checkpoint (kept outside git, see below); evaluating a
model needs no training. On the dev ground truth (real recordings used as "model output", Track A) the practical ceilings are:

| metric | real speech |
|---|---|
| NVPA (tolerance 0:1, thresholds ×0.75) | 0.850 (95% CI by speaker [0.766, 0.881]); random placement 0.135 |
| WER | 0.027 |
| pMOS (DNSMOS `ovrl`) | 3.13 (0.533 normalised) |
| Speaker similarity (cosine) | 0.810 |
| AutoScore (max 0.70, not official) | 0.562 |

### Quick start: evaluate a model on Kaggle (no training)
Open `notebooks/kaggle_nvtts_eval.ipynb` (File → Import Notebook), then:
1. Add inputs: the dataset with `vinv-tts/{train,dev}`, the checkpoint dataset (`nvpa_gap.joblib` + `summary.json` of the real-speech run),
   and a dataset with **your model's output for the 316 dev sentences**, one file per sample named `<spk_id>_<audio stem>.wav`
   (e.g. `spk_0000_0001.wav`). `dev_inputs.jsonl`, written by the notebook, lists the tagged texts to synthesize.
2. Settings: Internet **On**; GPU optional.
3. Edit `GEN` and `RUN_NAME` in step 1 and run all cells (about 10 minutes).

The notebook builds the configuration from the checkpoint's `summary.json`, so normalization, tolerance and thresholds are identical to
the real-speech run; it stops if any sample is missing (missing samples would be excluded from the metrics and make scores
incomparable). It prints your model next to real speech, per-NV-type scores, per-sample diagnostics and saves a CSV and a zip.

### Command line
    pip install -e ".[audio,config,asr,mos,asv,nvpa]"       # install torch first for your CUDA/CPU setup; add ,dev for tests
    python scripts/make_tokens.py --bpe <asr dir>/bpe.model --out <asr dir>/tokens.txt    # once: the ASR repo ships no tokens.txt

    # manifest: dev texts paired with your outputs (<generated dir>/<sample_id>.wav)
    python -m nvtts_eval.data.adapter_vinv --root <data> --split dev --track A --source model --generated-dir <outputs> --out manifests/mymodel_A.jsonl
    python -m nvtts_eval.cli run --manifest manifests/mymodel_A.jsonl --run-dir runs/mymodel_A --config <config.yaml>
    python -m nvtts_eval.cli summary --manifest manifests/mymodel_A.jsonl --run-dir runs/mymodel_A --config <config.yaml>

`<config.yaml>`: start from `configs/default.yaml` and set `asr.model_dir`, `ss.savedir` and `nvpa.detector_path`; or copy
`active_assumptions` from the checkpoint's `summary.json` (what the notebook does). Results are cached per metric: unchanged inputs are
never recomputed, `--force` recomputes, `--limit N` with its own `--run-dir` is a quick smoke test. In PowerShell write each command on one line.

Reading a report:
* **NVPA** is the share of required NV events found at the right place with the right type. Read it with the *random-placement
  baseline*, the **spurious NVs per 100 words** (a system that inserts NVs everywhere scores high on NVPA) and the **per-type** values:
  breathing is 83% of events, sniff and throat clearing have very few (low `n`, wide intervals).
* **AutoScore** uses the official weights of NVPA, WER, pMOS and SS only, so its maximum is 0.70; it is never the official score.
  The **final score** needs human SN and Q (below). Failed or missing samples are listed and excluded, never silently dropped.
* Two confidence intervals are printed: by utterance, and by speaker (wider, because a few speakers dominate the data).

Track B (unseen speaker) needs `--track B --reference-map refs.json` (`{sample_id: [reference clips relative to --root]}`): the released
data contains no unseen speakers or reference-clip field.

### Human evaluation (adds SN, Q and the final score)
    python -m nvtts_eval.cli human-subset --manifest <m> --run-dir <run>                      # stratified, deterministic subset
    python -m nvtts_eval.cli human-export --manifest <m> --run-dir <run> --out <package dir>
    #   give raters ONLY <package dir>/for_raters; each rater fills a copy of rating_sheet.csv and returns it as <rater>.csv
    python -m nvtts_eval.cli human-import --manifest <m> --run-dir <run> --package <package dir> --ratings alice.csv bob.csv
Hidden anchors (real and noise-degraded audio) flag raters whose scale is off; keep `PRIVATE_key.json` away from raters. If raters also
fill the optional `NV_placement` column, the report prints its Spearman correlation with NVPA per sample: the best check that the NVPA
detector can be trusted on synthetic speech. The final score uses the automatic metrics on the rated subset.

### The calibration checkpoint (already produced; redo only if you change the detector)
* `nvpa_gap.joblib`: window-level NV detector (gradient boosting on hand-crafted acoustic features), trained on the **train** split only;
  `summary.json`: the real-speech run (dev ground truth) that defines the ceilings and the configuration.
* Not in git (`*.joblib` is ignored): keep it as a private dataset. It was trained with scikit-learn 1.6.1; loading it with another
  minor version prints a warning (install the same version or retrain).
* To redo: build manifests (`--split train --no-references`), `train-detector --manifest train.jsonl --run-dir runs/train_gt --out
  nvpa_gap.joblib`, run the dev ground truth, then `nvpa-sweep --run-dir <run> --tolerances 0 1 0:1 1:0 2 --scales 0.5 0.75 1.0` and pick
  the row with a high lift (NVPA minus random placement) and a spurious rate near the gold NV density (5.4 per 100 words).

### Layout and development
    nvtts_eval/   data (parser, manifest, adapter) · metrics (wer, pmos, speaker_sim) · models (ASR/DNSMOS/ECAPA wrappers)
                  nvpa (windows, features, detector, matching, sweep) · human · report · core (cache) · cli
    configs/ scripts/ notebooks/ tests/ (160 tests: python -m pytest -q)
ASR weights are CC BY-NC-ND 4.0: use locally, never commit them. DNSMOS loads remote code (`trust_remote_code`): pin `pmos.revision`.
Not implemented yet: adapters for the organizers' public/private test formats (not released), a local Track B protocol.

---

## Tiếng Việt

Công cụ đánh giá **một hệ thống TTS** cho cuộc thi **ViNV-TTS** (VLSP 2026: giọng nói hội thoại tiếng Việt có âm phi lời). Nó chấm giọng tổng hợp
bằng WER (ASR Zipformer), MOS dự đoán (DNSMOS), độ giống giọng (ECAPA) và NVPA (xấp xỉ của độ chính xác vị trí NV, dùng khi phát triển), rồi kết hợp
với điểm người chấm (SN, Q) theo công thức của cuộc thi. Đây **không phải** bộ chấm chính thức của ban tổ chức: mọi chỗ đề bài chưa quy định
đều là `[ASSUMPTION]`, chỉnh được và được in trong mọi báo cáo. Thiết kế và số đo: `NVTTS-Eval-Spec-v2.md`.

### Trạng thái
Phần **hiệu chuẩn đã xong**. Detector và điểm vận hành nằm trong một checkpoint (để ngoài git, xem dưới); đánh giá một model **không cần huấn luyện gì**.
Trên dev ground truth (ghi âm thật đóng vai "output của model", Track A), các mức tối đa thực tế là:

| chỉ số | giọng thật |
|---|---|
| NVPA (dung sai 0:1, ngưỡng ×0.75) | 0.850 (CI 95% theo speaker [0.766, 0.881]); vị trí ngẫu nhiên 0.135 |
| WER | 0.027 |
| pMOS (DNSMOS `ovrl`) | 3.13 (0.533 sau chuẩn hóa) |
| Độ giống giọng (cosine) | 0.810 |
| AutoScore (tối đa 0.70, không chính thức) | 0.562 |

### Bắt đầu nhanh: đánh giá model trên Kaggle (không huấn luyện)
Mở `notebooks/kaggle_nvtts_eval.ipynb` (File → Import Notebook), rồi:
1. Add Input: dataset dữ liệu `vinv-tts/{train,dev}`, dataset checkpoint (`nvpa_gap.joblib` và `summary.json` của lần chạy giọng thật), và dataset chứa
   **output của model cho 316 câu dev**, mỗi mẫu một file đặt tên `<spk_id>_<tên file audio>.wav` (ví dụ `spk_0000_0001.wav`). Notebook ghi sẵn
   `dev_inputs.jsonl` (văn bản có thẻ NV) để bạn đưa vào model sinh audio.
2. Settings: bật Internet; GPU là tùy chọn.
3. Sửa `GEN` và `RUN_NAME` ở bước 1 rồi chạy toàn bộ (khoảng 10 phút).

Notebook tạo cấu hình từ `summary.json` trong checkpoint nên chuẩn hóa, dung sai, ngưỡng giống hệt lần chạy giọng thật; nó **dừng nếu thiếu mẫu**
(mẫu thiếu bị loại khỏi chỉ số, làm điểm không còn so sánh được). Kết quả in ra bên cạnh giọng thật, kèm điểm theo từng loại NV, chi tiết từng
mẫu, và lưu CSV cùng file zip.

### Dòng lệnh
    pip install -e ".[audio,config,asr,mos,asv,nvpa]"       # cài torch trước theo CUDA/CPU của máy; thêm ,dev để chạy test
    python scripts/make_tokens.py --bpe <thư mục ASR>/bpe.model --out <thư mục ASR>/tokens.txt    # một lần: repo ASR không có tokens.txt

    # manifest: ghép văn bản dev với output của bạn (<thư mục output>/<sample_id>.wav)
    python -m nvtts_eval.data.adapter_vinv --root <data> --split dev --track A --source model --generated-dir <output> --out manifests/mymodel_A.jsonl
    python -m nvtts_eval.cli run --manifest manifests/mymodel_A.jsonl --run-dir runs/mymodel_A --config <config.yaml>
    python -m nvtts_eval.cli summary --manifest manifests/mymodel_A.jsonl --run-dir runs/mymodel_A --config <config.yaml>

`<config.yaml>`: bắt đầu từ `configs/default.yaml` rồi đặt `asr.model_dir`, `ss.savedir`, `nvpa.detector_path`; hoặc chép khối `active_assumptions` trong
`summary.json` của checkpoint (notebook làm đúng như vậy). Kết quả được cache theo từng chỉ số: đầu vào không đổi thì không tính lại,
`--force` để tính lại, `--limit N` kèm `--run-dir` riêng để chạy thử nhanh. Trong PowerShell viết mỗi lệnh trên một dòng.

Cách đọc báo cáo:
* **NVPA** là tỷ lệ NV bắt buộc được tìm thấy đúng loại, đúng vị trí. Đọc kèm *mức vị trí ngẫu nhiên*, **NV thừa trên 100 từ** (hệ thống chèn NV
  tràn lan sẽ được NVPA cao) và điểm **theo từng loại**: breathing chiếm 83% sự kiện, sniff và throatclearing rất ít (`n` nhỏ, khoảng tin cậy rộng).
* **AutoScore** chỉ dùng trọng số chính thức của NVPA, WER, pMOS, SS nên tối đa 0.70, và không bao giờ là điểm chính thức. **Điểm cuối** cần SN và Q
  của người chấm (bên dưới). Mẫu lỗi hoặc thiếu được liệt kê và loại ra, không bị bỏ âm thầm.
* Có hai khoảng tin cậy: theo câu và theo speaker (rộng hơn vì vài speaker chiếm phần lớn dữ liệu).

Track B (speaker lạ) cần `--track B --reference-map refs.json` (`{sample_id: [clip tham chiếu, đường dẫn tương đối so với --root]}`): dữ liệu công bố
không có speaker lạ hay trường clip tham chiếu.

### Đánh giá bởi người (thêm SN, Q và điểm cuối)
    python -m nvtts_eval.cli human-subset --manifest <m> --run-dir <run>                      # tập con phân tầng, cố định theo seed
    python -m nvtts_eval.cli human-export --manifest <m> --run-dir <run> --out <thư mục gói>
    #   chỉ gửi người chấm <thư mục gói>/for_raters; mỗi người điền một bản rating_sheet.csv và gửi lại đặt tên <người chấm>.csv
    python -m nvtts_eval.cli human-import --manifest <m> --run-dir <run> --package <thư mục gói> --ratings alice.csv bob.csv
Các mẫu neo ẩn (audio thật và audio bị làm nhiễu) giúp phát hiện người chấm lệch thang điểm; giữ kín `PRIVATE_key.json`. Nếu người chấm điền thêm cột
tùy chọn `NV_placement`, báo cáo in tương quan Spearman giữa điểm đó và NVPA theo từng mẫu: phép kiểm chứng tốt nhất rằng detector NVPA đáng tin trên
giọng tổng hợp. Điểm cuối dùng các chỉ số tự động tính trên đúng tập con đã chấm.

### Checkpoint hiệu chuẩn (đã tạo xong; chỉ làm lại nếu đổi detector)
* `nvpa_gap.joblib`: detector NV theo cửa sổ (gradient boosting trên đặc trưng âm học tự thiết kế), huấn luyện chỉ trên tập **train**;
  `summary.json`: kết quả lần chạy giọng thật (dev ground truth), xác định các mức tối đa và cấu hình.
* Không nằm trong git (`*.joblib` bị ignore): hãy giữ ở một dataset private. Detector huấn luyện bằng scikit-learn 1.6.1; nạp bằng phiên bản minor khác
  sẽ in cảnh báo (cài cùng phiên bản hoặc huấn luyện lại).
* Cách làm lại: tạo manifest (`--split train --no-references`), `train-detector --manifest train.jsonl --run-dir runs/train_gt --out nvpa_gap.joblib`,
  chạy dev ground truth, rồi `nvpa-sweep --run-dir <run> --tolerances 0 1 0:1 1:0 2 --scales 0.5 0.75 1.0` và chọn dòng có lift cao (NVPA trừ mức ngẫu nhiên)
  với NV thừa gần mật độ NV thật (5.4 trên 100 từ).

### Cấu trúc và phát triển
    nvtts_eval/   data (parser, manifest, adapter) · metrics (wer, pmos, speaker_sim) · models (bọc ASR/DNSMOS/ECAPA)
                  nvpa (cửa sổ, đặc trưng, detector, so khớp, sweep) · human · report · core (cache) · cli
    configs/ scripts/ notebooks/ tests/ (160 test: python -m pytest -q)
Trọng số ASR theo giấy phép CC BY-NC-ND 4.0: chỉ dùng cục bộ, không commit. DNSMOS nạp code từ xa (`trust_remote_code`): nên ghim `pmos.revision`.
Chưa có: adapter cho định dạng public/private test của ban tổ chức (chưa công bố), quy trình Track B cục bộ.

# Linux

Quy ước: dữ liệu, model và checkpoint nằm dưới `/data`.

## 1. Môi trường

Cần Python ≥ 3.9, `git` và `libsndfile1` (để đọc FLAC).

```bash
sudo apt install -y libsndfile1 git
git clone https://github.com/VietCH57/NVTTS-Evaluator.git && cd NVTTS-Evaluator
python3 -m venv .venv && source .venv/bin/activate
pip install torch                       # có GPU: cài bản CUDA khớp driver (pytorch.org); không có GPU vẫn chạy được
pip install -e ".[audio,config,asr,mos,asv,nvpa]"
pip install scikit-learn==1.6.1         # khớp phiên bản lúc huấn luyện detector, tránh cảnh báo và sai lệch
```

## 2. Đưa dữ liệu và checkpoint lên server

| Thành phần | Đường dẫn ví dụ | Ghi chú |
|---|---|---|
| Dữ liệu | `/data/vinv-tts/{train,dev}` | Cần cả hai: `dev` cho văn bản, `train` cho clip tham chiếu của SS |
| Checkpoint | `/data/ckpt/` | Chứa `nvpa_gap.joblib` và `summary.json` |
| Output của model | `/data/outputs/mymodel/` | Mỗi mẫu một file `<spk_id>_<tên file audio>.wav`, ví dụ `spk_0000_0001.wav` |

## 3. Tải ASR (một lần)

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("hynt/Zipformer-30M-RNNT-6000h",
    allow_patterns=["encoder-epoch-20-avg-10.onnx", "decoder-epoch-20-avg-10.onnx",
                    "joiner-epoch-20-avg-10.onnx", "bpe.model"],
    local_dir="/data/models/Zipformer-30M-RNNT-6000h")
EOF
python scripts/make_tokens.py \
    --bpe /data/models/Zipformer-30M-RNNT-6000h/bpe.model \
    --out /data/models/Zipformer-30M-RNNT-6000h/tokens.txt
```

## 4. Tạo cấu hình từ checkpoint

Cấu hình được sao từ `summary.json` của lần chạy giọng thật (chuẩn hóa, dung sai, ngưỡng), chỉ đổi đường dẫn. Nhờ vậy kết quả model so sánh được với mức tối đa của giọng thật.

```bash
python - <<'EOF'
import json, yaml
cfg = json.load(open("/data/ckpt/summary.json"))["active_assumptions"]
cfg["asr"]["model_dir"] = "/data/models/Zipformer-30M-RNNT-6000h"
cfg["ss"]["savedir"] = "/data/models/spkrec-ecapa-voxceleb"
cfg["nvpa"]["detector_path"] = "/data/ckpt/nvpa_gap.joblib"
yaml.safe_dump(cfg, open("/data/eval.yaml", "w"), sort_keys=False)
EOF
```

## 5. Chạy

```bash
python -m nvtts_eval.data.adapter_vinv --root /data/vinv-tts --split dev --track A --source model \
    --generated-dir /data/outputs/mymodel --out manifests/mymodel_A.jsonl

nohup python -m nvtts_eval.cli run --manifest manifests/mymodel_A.jsonl \
    --run-dir runs/mymodel_A --config /data/eval.yaml > run.log 2>&1 &
```

- Kết quả được cache theo từng chỉ số: chạy lại sẽ tiếp tục, không tính lại phần đã xong.
- Báo cáo nằm ở `runs/mymodel_A/summary.json`.
- Xem lại báo cáo sau đó: chạy lệnh `summary` với cùng `--manifest`, `--run-dir`, `--config`.

## Lưu ý

- **Server không có internet.** DNSMOS (kèm code từ xa) và ECAPA tự tải từ Hugging Face ở lần chạy đầu. Chạy một lần trên máy có mạng, rồi chép `~/.cache/huggingface` và `/data/models/spkrec-ecapa-voxceleb` sang server và đặt `export HF_HUB_OFFLINE=1`. Nên ghim `pmos.revision` vào một commit cụ thể, sau khi đã đọc code của repo DNSMOS.
- **Tốc độ.** ASR luôn chạy trên CPU: tăng `asr.num_threads` trong YAML nếu server có nhiều lõi. GPU chỉ tăng tốc DNSMOS và ECAPA.
- **Chạy nhiều lần cùng lúc.** Mỗi lần dùng một `--run-dir` riêng, không chia sẻ thư mục.