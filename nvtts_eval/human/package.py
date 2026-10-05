"""Anonymised rating package.

    <out>/for_raters/audio/item_0001.wav ...   audio, anonymous names, original sample rate, PCM16
    <out>/for_raters/rating_sheet.csv           one row per item; raters fill SN, Q (+ optional NV columns)
    <out>/for_raters/README.md                  instructions (Vietnamese)
    <out>/PRIVATE_key.json                      item -> sample / kind. DO NOT give this to raters.

Hidden anchors (spec v2): ground-truth items and artificially degraded items are mixed into the
list. They let us check each rater's scale: ground truth should be rated clearly above degraded audio.
Item order is shuffled with a fixed seed.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..audio import load_mono
from ..data.manifest import Manifest, Sample

SHEET_COLUMNS = ["item_id", "audio_file", "transcript", "SN", "Q", "NV_naturalness", "NV_placement", "comment"]

README_VI = """# Hướng dẫn chấm điểm

Bạn sẽ nghe các file âm thanh trong thư mục `audio/` và điền điểm vào file `rating_sheet.csv`
(mở bằng Excel hoặc Google Sheets, giữ nguyên tên cột). Mỗi dòng là một file. Thang điểm từ 1 đến 5
(có thể dùng số thập phân, ví dụ 3.5). Cột `transcript` là văn bản đầu vào; thẻ như `[laughter]`
cho biết tiếng phi lời nào phải xuất hiện, ở chỗ nào.

Hai cột bắt buộc:

* **SN – Độ tự nhiên của lời nói**: giọng nói nghe có tự nhiên, trôi chảy, giống người thật không,
  kể cả phần lời nói quanh các tiếng phi lời. 1 = rất giả/máy móc, 5 = như người thật.
* **Q – Chất lượng âm thanh**: có nhiễu, méo, rè, cắt cụt, tạp âm, lỗi kỹ thuật không.
  1 = rất tệ, 5 = sạch hoàn toàn.

Hai cột tùy chọn (điền nếu được yêu cầu):

* **NV_naturalness**: các tiếng phi lời (cười, thở, hít mũi, hắng giọng) nghe có tự nhiên, hòa vào lời nói không.
* **NV_placement**: các tiếng phi lời có đúng loại và đúng vị trí như thẻ trong `transcript` không.

Lưu ý: chấm độc lập từng file, không cần so sánh với file khác; nghe trong môi trường yên tĩnh, cùng
một thiết bị (tai nghe) cho tất cả; không chia sẻ nội dung file này.
"""


def degrade_snr(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at the given SNR (dB). Deterministic for a seeded generator."""
    p = float(np.mean(wav.astype(np.float64) ** 2)) or 1e-12
    noise = rng.standard_normal(len(wav)) * np.sqrt(p / (10 ** (snr_db / 10)))
    return np.clip(wav + noise, -1.0, 1.0).astype(np.float32)


def _write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def export_package(manifest: Manifest, sample_ids: Sequence[str], out_dir: Path, n_gt_anchors: int = 5,
                   n_degraded_anchors: int = 5, snr_db: float = 5.0, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    rater_dir = out_dir / "for_raters"
    (rater_dir / "audio").mkdir(parents=True, exist_ok=True)

    wanted = set(sample_ids)
    by_id = {s.sample_id: s for s in manifest.samples}
    model_items: List[Sample] = [by_id[i] for i in sorted(wanted) if i in by_id]
    skipped = [s.sample_id for s in model_items if not (manifest.resolve_generated(s) and manifest.resolve_generated(s).is_file())]
    model_items = [s for s in model_items if s.sample_id not in skipped]

    gt_pool = [s for s in manifest.samples if manifest.resolve_ground_truth(s) is not None
               and manifest.resolve_ground_truth(s).is_file()]
    outside = [s for s in gt_pool if s.sample_id not in wanted] or gt_pool
    rng.shuffle(outside)
    n_gt = min(n_gt_anchors, len(outside))
    gt_anchors = outside[:n_gt]
    deg_pool = [s for s in outside[n_gt:]] or outside
    degraded = deg_pool[:min(n_degraded_anchors, len(deg_pool))]

    items = [("model", s) for s in model_items] + [("anchor_gt", s) for s in gt_anchors] + \
            [("anchor_degraded", s) for s in degraded]
    rng.shuffle(items)

    key: List[Dict[str, Any]] = []
    rows: List[Dict[str, str]] = []
    for n, (kind, s) in enumerate(items, 1):
        item_id = f"item_{n:04d}"
        src = manifest.resolve_generated(s) if kind == "model" else manifest.resolve_ground_truth(s)
        wav, sr = load_mono(src)
        if kind == "anchor_degraded":
            wav = degrade_snr(wav, snr_db, nrng)
        _write_wav(rater_dir / "audio" / f"{item_id}.wav", wav, sr)
        key.append({"item_id": item_id, "kind": kind, "sample_id": s.sample_id, "speaker_id": s.speaker_id})
        rows.append({"item_id": item_id, "audio_file": f"audio/{item_id}.wav", "transcript": s.text,
                     "SN": "", "Q": "", "NV_naturalness": "", "NV_placement": "", "comment": ""})
    with open(rater_dir / "rating_sheet.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (rater_dir / "README.md").write_text(README_VI, encoding="utf-8")
    meta = {"items": key, "seed": seed, "snr_db": snr_db, "skipped_missing_audio": skipped,
            "note": "PRIVATE: maps anonymous items to samples and anchor kinds. Do not share with raters."}
    (out_dir / "PRIVATE_key.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
