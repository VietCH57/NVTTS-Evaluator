#!/usr/bin/env python3
"""Thống kê dataset ViNV-TTS (Phase 1: dataset understanding).

Cấu trúc dữ liệu đã biết (từ người dùng):
    <root>/<split>/<spk_id>/<spk_id>.json   # list[{"audio", "text", "language_id"}]
    <root>/<split>/<spk_id>/<audio files>
    split in {train, dev}; spk_id giống nhau giữa train và dev là cùng một người.

Cách dùng:
    python dataset_stats.py --root D:\\Work\\NVTTS-Evaluator\\data\\vinv-tts --out stats_out
    python dataset_stats.py --root ... --no-audio      # bỏ qua đọc header audio

Chỉ phụ thuộc stdlib; nếu có `soundfile` thì đọc thêm thông tin audio (chỉ đọc header, nhanh).
Script này CHỈ mô tả dữ liệu, không đưa ra giả định nào cho evaluator.
Regex tag ở đây tạm thời, sau này sẽ thay bằng parser NV chuẩn của codebase.
"""
import argparse
import csv
import json
import math
import re
import string
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

TAG_RE = re.compile(r"\[[^\[\]]*\]")
KNOWN_TAGS = ["[laughter]", "[breathing]", "[sniff]", "[throatclearing]"]
AUDIO_EXT = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}
PUNCT_RE = re.compile(r"[.,?!;:…\"“”()]")
# Chữ cái hợp lệ: a-z (kể cả f, j, w, z vì có trong từ mượn) + ký tự có dấu tiếng Việt.
VI_LETTERS = set(string.ascii_lowercase) | set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
)


# ----------------------------------------------------------------------------- helpers
def percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def describe(vals):
    v = sorted(x for x in vals if x is not None)
    if not v:
        return {}
    mean = sum(v) / len(v)
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / len(v))
    d = {"n": len(v), "mean": mean, "std": std}
    for q in (0, 5, 25, 50, 75, 95, 100):
        d[f"p{q}"] = percentile(v, q)
    return d


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def table(rows, headers, title=None):
    rows = [[fmt(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]
    if title:
        print(f"\n=== {title} ===")
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def print_describe(name, d, nd=2):
    if not d:
        print(f"  {name}: (không có dữ liệu)")
        return
    print(f"  {name}: n={d['n']} mean={d['mean']:.{nd}f} std={d['std']:.{nd}f} "
          f"min={d['p0']:.{nd}f} p5={d['p5']:.{nd}f} p25={d['p25']:.{nd}f} "
          f"med={d['p50']:.{nd}f} p75={d['p75']:.{nd}f} p95={d['p95']:.{nd}f} max={d['p100']:.{nd}f}")


def get_audio_info(path):
    try:
        import soundfile as sf
    except ImportError:
        return None
    try:
        i = sf.info(str(path))
        return {"sr": i.samplerate, "ch": i.channels, "dur": i.frames / i.samplerate,
                "fmt": i.format, "subtype": i.subtype}
    except Exception as e:  # file hỏng
        return {"error": str(e)}


# ----------------------------------------------------------------------------- analysis
def analyze_text(text):
    """Trả về các thuộc tính mô tả của một transcript (không giả định gì về chuẩn)."""
    tags = list(TAG_RE.finditer(text))
    clean = " ".join(TAG_RE.sub(" ", text).split())
    n_words = len(clean.split()) if clean else 0

    events = []  # (type, gap_index) với gap_index = số từ đứng trước tag
    for m in tags:
        before = " ".join(TAG_RE.sub(" ", text[: m.start()]).split())
        events.append((m.group(0), len(before.split()) if before else 0))

    def pos_class(g):
        if n_words == 0:
            return "only"
        return "start" if g == 0 else "end" if g == n_words else "mid"

    flags = {
        "has_upper": text != text.lower(),
        "has_digit": bool(re.search(r"\d", text)),
        "has_punct": bool(PUNCT_RE.search(text)),
        "ws_issue": text != text.strip() or "  " in text,
        "tag_glued": bool(re.search(r"[^\s\[]\[|\][^\s\]]", text)),   # tag dính liền ký tự khác
        "tag_next_to_punct": bool(re.search(r"\][.,?!;:]|[.,?!;:]\s*\[", text)),
        "unbalanced_bracket": text.count("[") != text.count("]") or bool(
            re.sub(r"\[[^\[\]]*\]", "", text).count("[") or re.sub(r"\[[^\[\]]*\]", "", text).count("]")),
        "not_nfc": text != unicodedata.normalize("NFC", text),
        "non_vi_chars": sorted({c for c in clean.lower() if c.isalpha() and c not in VI_LETTERS})[:10],
    }
    gaps = [g for _, g in events]
    return {
        "clean_text": clean, "n_words": n_words, "events": events,
        "pos_classes": [pos_class(g) for g in gaps],
        "rel_pos": [g / n_words if n_words else 0.0 for g in gaps],
        "n_same_gap": sum(c - 1 for c in Counter(gaps).values() if c > 1),  # tag liền nhau
        "flags": flags,
    }


def load_split(root, split, use_audio):
    split_dir = root / split
    rows, problems = [], []
    if not split_dir.is_dir():
        problems.append(f"Không thấy thư mục split: {split_dir}")
        return rows, problems
    for spk_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        meta = spk_dir / f"{spk_dir.name}.json"
        if not meta.exists():
            cands = list(spk_dir.glob("*.json"))
            if len(cands) != 1:
                problems.append(f"[{split}/{spk_dir.name}] không tìm thấy metadata json duy nhất")
                continue
            meta = cands[0]
        try:
            items = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"[{split}/{spk_dir.name}] lỗi đọc json: {e}")
            continue
        listed = set()
        for it in items:
            t = analyze_text(it.get("text", ""))
            audio = it.get("audio", "")
            listed.add(audio)
            apath = spk_dir / audio
            info = get_audio_info(apath) if use_audio and apath.exists() else None
            rows.append({
                "split": split, "speaker": spk_dir.name, "audio": audio,
                "text": it.get("text", ""), "language_id": it.get("language_id"),
                "extra_keys": sorted(set(it) - {"audio", "text", "language_id"}),
                "audio_exists": apath.exists(), "audio_info": info, **t,
            })
        for f in spk_dir.iterdir():
            if f.suffix.lower() in AUDIO_EXT and f.name not in listed:
                problems.append(f"[{split}/{spk_dir.name}] audio không có trong metadata: {f.name}")
    return rows, problems


# ----------------------------------------------------------------------------- report
def report(rows_by_split, problems, out_dir, use_audio):
    summary = {}
    all_rows = [r for rs in rows_by_split.values() for r in rs]
    has_audio = use_audio and any(r["audio_info"] and "dur" in r["audio_info"] for r in all_rows)

    def dur(r):
        i = r["audio_info"]
        return i["dur"] if i and "dur" in i else None

    # 1. Tổng quan theo split
    ov = []
    for sp, rs in rows_by_split.items():
        durs = [dur(r) for r in rs if dur(r) is not None]
        n_nv = sum(len(r["events"]) for r in rs)
        ov.append([sp, len({r["speaker"] for r in rs}), len(rs), sum(r["n_words"] for r in rs),
                   n_nv, sum(1 for r in rs if not r["events"]),
                   (sum(durs) / 3600) if durs else None])
    table(ov, ["split", "speakers", "utts", "words", "NV events", "utts không NV", "hours"],
          "1. Tổng quan")
    summary["overview"] = ov

    # 2. Speaker & overlap
    spk = defaultdict(lambda: defaultdict(lambda: {"utts": 0, "dur": 0.0, "nv": Counter()}))
    for r in all_rows:
        s = spk[r["speaker"]][r["split"]]
        s["utts"] += 1
        s["dur"] += dur(r) or 0.0
        s["nv"].update(t for t, _ in r["events"])
    splits = list(rows_by_split)
    spk_rows = []
    for sid in sorted(spk):
        row = [sid]
        for sp in splits:
            s = spk[sid].get(sp)
            row += [s["utts"] if s else 0, (s["dur"] / 60) if s and has_audio and s["dur"] else None]
        tr = spk[sid].get("train")
        row += [tr["nv"][t] if tr else 0 for t in KNOWN_TAGS]
        spk_rows.append(row)
    hdr = ["speaker"] + [f"{sp}_{k}" for sp in splits for k in ("utts", "min")] + \
          [f"train{t}" for t in KNOWN_TAGS]
    table(spk_rows, hdr, "2. Theo speaker")
    sets = {sp: {r["speaker"] for r in rs} for sp, rs in rows_by_split.items()}
    if "train" in sets and "dev" in sets:
        print(f"\n  speaker chỉ có ở train: {len(sets['train'] - sets['dev'])} | "
              f"có ở cả hai: {len(sets['train'] & sets['dev'])} | "
              f"chỉ có ở dev: {len(sets['dev'] - sets['train'])}")
        only_dev = sorted(sets["dev"] - sets["train"])
        if only_dev:
            print(f"  CẢNH BÁO speaker chỉ có ở dev: {only_dev}")
    with open(out_dir / "speakers.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(spk_rows)

    # 3. Kho tag
    inv = {sp: Counter(t for r in rs for t, _ in r["events"]) for sp, rs in rows_by_split.items()}
    tags_all = sorted(set().union(*inv.values()), key=lambda t: (t not in KNOWN_TAGS, t))
    table([[t, "known" if t in KNOWN_TAGS else "LẠ"] + [inv[sp][t] for sp in splits] for t in tags_all],
          ["tag", "status"] + splits, "3. Kho tag")
    missing = [t for t in KNOWN_TAGS if all(inv[sp][t] == 0 for sp in splits)]
    if missing:
        print(f"  Tag trong đề bài nhưng KHÔNG xuất hiện: {missing}")

    # 4. Theo loại NV: tần suất, vị trí
    for sp, rs in rows_by_split.items():
        total_words = sum(r["n_words"] for r in rs)
        total_min = sum(dur(r) or 0 for r in rs) / 60
        rows = []
        for t in tags_all:
            n = inv[sp][t]
            if n == 0 and t not in KNOWN_TAGS:
                continue
            n_utt = sum(1 for r in rs if any(x == t for x, _ in r["events"]))
            pc = Counter(p for r in rs for (x, _), p in zip(r["events"], r["pos_classes"]) if x == t)
            rows.append([t, n, n_utt, 100 * n_utt / len(rs) if rs else 0,
                         100 * n / total_words if total_words else 0,
                         n / total_min if has_audio and total_min else None,
                         pc["start"], pc["mid"], pc["end"]])
        table(rows, ["tag", "count", "utts", "%utts", "per100words", "per_min",
                     "@start", "@mid", "@end"], f"4. Theo loại NV [{sp}]")

    # 5. Số NV / câu, vị trí tương đối, tag liền nhau
    for sp, rs in rows_by_split.items():
        print(f"\n=== 5. Cấu trúc NV [{sp}] ===")
        hist = Counter(len(r["events"]) for r in rs)
        print("  số NV/câu:", {k: hist[k] for k in sorted(hist)})
        rel = Counter(min(int(p * 10), 9) for r in rs for p in r["rel_pos"])
        print("  vị trí tương đối (decile 0=đầu .. 9=cuối):", {k: rel[k] for k in sorted(rel)})
        print("  số NV đứng liền nhau (cùng một khe giữa hai từ):",
              sum(r["n_same_gap"] for r in rs))
        mixed = Counter(len({t for t, _ in r["events"]}) for r in rs if r["events"])
        print("  số loại NV khác nhau/câu:", {k: mixed[k] for k in sorted(mixed)})

    # 6. Độ dài, tốc độ nói
    summary["length"] = {}
    for sp, rs in rows_by_split.items():
        print(f"\n=== 6. Độ dài [{sp}] ===")
        words = [r["n_words"] for r in rs]
        print_describe("số từ/câu", describe(words), 1)
        summary["length"][sp] = {"words": describe(words)}
        if has_audio:
            durs = [dur(r) for r in rs if dur(r) is not None]
            print_describe("thời lượng (s)", describe(durs))
            rate = [r["n_words"] / dur(r) for r in rs if dur(r)]
            print_describe("từ/giây", describe(rate))
            nvd = [len(r["events"]) / dur(r) for r in rs if dur(r)]
            print_describe("NV/giây", describe(nvd))
            summary["length"][sp].update({"dur": describe(durs), "words_per_sec": describe(rate)})
            sr = Counter(r["audio_info"]["sr"] for r in rs if r["audio_info"] and "sr" in r["audio_info"])
            ch = Counter(r["audio_info"]["ch"] for r in rs if r["audio_info"] and "ch" in r["audio_info"])
            fm = Counter((r["audio_info"]["fmt"], r["audio_info"]["subtype"]) for r in rs
                         if r["audio_info"] and "fmt" in r["audio_info"])
            print(f"  sample rate: {dict(sr)} | channels: {dict(ch)} | format: {dict(fm)}")

    # 7. Chất lượng text/metadata
    qrows = []
    keys = ["has_upper", "has_digit", "has_punct", "ws_issue", "tag_glued",
            "tag_next_to_punct", "unbalanced_bracket", "not_nfc"]
    for sp, rs in rows_by_split.items():
        qrows.append([sp] + [sum(1 for r in rs if r["flags"][k]) for k in keys] +
                     [sum(1 for r in rs if r["flags"]["non_vi_chars"])] +
                     [sum(1 for r in rs if r["n_words"] == 0)])
    table(qrows, ["split"] + keys + ["non_vi_chars", "empty_text"], "7. Chất lượng transcript (số câu dính cờ)")
    langs = Counter(r["language_id"] for r in all_rows)
    extra = Counter(k for r in all_rows for k in r["extra_keys"])
    print(f"  language_id: {dict(langs)} | key lạ trong metadata: {dict(extra) or 'không có'}")
    ex = [r for r in all_rows if r["flags"]["non_vi_chars"]][:5]
    for r in ex:
        print(f"  ví dụ ký tự ngoài tiếng Việt {r['flags']['non_vi_chars']}: "
              f"{r['split']}/{r['speaker']}/{r['audio']}")

    # 8. Audio và rò rỉ train->dev
    miss = [r for r in all_rows if not r["audio_exists"]]
    bad = [r for r in all_rows if r["audio_info"] and "error" in r["audio_info"]]
    print(f"\n=== 8. Audio & trùng lặp ===\n  thiếu file audio: {len(miss)} | audio lỗi: {len(bad)}")
    for r in (miss + bad)[:5]:
        print(f"    {r['split']}/{r['speaker']}/{r['audio']}")
    if "train" in rows_by_split and "dev" in rows_by_split:
        tr_text = {r["clean_text"] for r in rows_by_split["train"]}
        dup = [r for r in rows_by_split["dev"] if r["clean_text"] in tr_text]
        print(f"  câu dev trùng clean_text với train: {len(dup)}/{len(rows_by_split['dev'])}")
    for sp, rs in rows_by_split.items():
        c = Counter(r["clean_text"] for r in rs)
        print(f"  [{sp}] clean_text lặp trong cùng split: {sum(v - 1 for v in c.values() if v > 1)}")

    # 9. Vấn đề phát hiện khi load
    print(f"\n=== 9. Vấn đề khi load: {len(problems)} ===")
    for p in problems[:20]:
        print("  ", p)

    # Ghi file
    with open(out_dir / "utterances.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["split", "speaker", "audio", "duration", "n_words", "n_nv", "nv_types",
                    "nv_gap_index", "nv_pos_class", "text", "clean_text"])
        for r in all_rows:
            w.writerow([r["split"], r["speaker"], r["audio"], dur(r), r["n_words"],
                        len(r["events"]), "|".join(t for t, _ in r["events"]),
                        "|".join(str(g) for _, g in r["events"]), "|".join(r["pos_classes"]),
                        r["text"], r["clean_text"]])
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nĐã ghi: {out_dir / 'utterances.csv'}, speakers.csv, summary.json")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="thư mục chứa train/ và dev/")
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--out", type=Path, default=Path("stats_out"))
    ap.add_argument("--no-audio", action="store_true", help="không đọc header audio")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    use_audio = not a.no_audio
    if use_audio:
        try:
            import soundfile  # noqa: F401
        except ImportError:
            print("Chưa cài soundfile -> bỏ qua thống kê audio (pip install soundfile)")
            use_audio = False
    rows_by_split, problems = {}, []
    for sp in a.splits:
        rows, pr = load_split(a.root, sp, use_audio)
        rows_by_split[sp] = rows
        problems += pr
    if not any(rows_by_split.values()):
        sys.exit(f"Không đọc được dữ liệu nào từ {a.root}")
    report(rows_by_split, problems, a.out, use_audio)


if __name__ == "__main__":
    main()
