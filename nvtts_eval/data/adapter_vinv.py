"""Adapter: released ViNV-TTS data -> canonical manifest.

This is the ONLY module that knows the released layout:

    <root>/<split>/<spk_id>/<spk_id>.json    # list of {"audio", "text", "language_id"}
    <root>/<split>/<spk_id>/<audio files>

Facts it relies on (verified on train/dev): speaker = folder name; metadata has no
timestamps, duration or reference-clip field; the same spk_id in train and dev is
the same person.

sample_id = "<spk_id>_<audio stem>", e.g. "spk_0000_0001". Generated audio for a
model run is expected at  <generated_dir>/<sample_id><ext>.

Reference audio (Track A)  [ASSUMPTION]: clips of the same speaker from
`reference_split` (default "train"), sorted by file name, optionally truncated to
`max_reference_clips`. Track B needs an explicit `reference_map`, because the
released data has no unseen speakers or reference-clip field.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .manifest import Manifest, ManifestHeader, Sample
from .nv_parser import NVParser


class DatasetFormatError(ValueError):
    pass


@dataclass(frozen=True)
class RawUtterance:
    split: str
    speaker_id: str
    audio_rel: str          # relative to dataset root, forward slashes
    text: str
    language_id: Optional[str]


def iter_split(root: Path, split: str) -> Iterator[RawUtterance]:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise DatasetFormatError(f"split directory not found: {split_dir}")
    for spk_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        meta = spk_dir / f"{spk_dir.name}.json"
        if not meta.exists():
            cands = sorted(spk_dir.glob("*.json"))
            if len(cands) != 1:
                raise DatasetFormatError(f"{spk_dir}: expected exactly one metadata json")
            meta = cands[0]
        items = json.loads(meta.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise DatasetFormatError(f"{meta}: expected a JSON list")
        for it in items:
            if not isinstance(it, dict) or "audio" not in it or "text" not in it:
                raise DatasetFormatError(f"{meta}: item without 'audio'/'text': {it!r}")
            yield RawUtterance(
                split=split,
                speaker_id=spk_dir.name,
                audio_rel=f"{split}/{spk_dir.name}/{it['audio']}",
                text=it["text"],
                language_id=it.get("language_id"),
            )


def sample_id_for(u: RawUtterance) -> str:
    return f"{u.speaker_id}_{Path(u.audio_rel).stem}"


@dataclass
class BuildReport:
    n_samples: int = 0
    n_missing_ground_truth: int = 0
    n_missing_generated: int = 0
    speakers_without_reference: List[str] = field(default_factory=list)
    language_ids: Counter = field(default_factory=Counter)

    def __str__(self) -> str:
        return (
            f"samples={self.n_samples} missing_ground_truth={self.n_missing_ground_truth} "
            f"missing_generated={self.n_missing_generated} "
            f"speakers_without_reference={len(self.speakers_without_reference)} "
            f"language_ids={dict(self.language_ids)}"
        )


def build_manifest(
    root: Path,
    split: str,
    track: str = "A",
    source: str = "ground_truth",            # "ground_truth" | "model"
    generated_dir: Optional[Path] = None,    # required when source == "model"
    generated_exts: Sequence[str] = (".wav", ".flac"),
    reference_split: str = "train",
    max_reference_clips: Optional[int] = None,
    reference_map: Optional[Dict[str, List[str]]] = None,  # sample_id -> paths rel. to root
    speakers: Optional[Sequence[str]] = None,
    parser: Optional[NVParser] = None,
) -> Tuple[Manifest, BuildReport]:
    root = Path(root)
    parser = parser or NVParser()
    if source not in ("ground_truth", "model"):
        raise ValueError("source must be 'ground_truth' or 'model'")
    if source == "model" and generated_dir is None:
        raise ValueError("source='model' requires generated_dir")
    if track == "B" and reference_map is None:
        raise ValueError(
            "Track B needs an explicit reference_map: the released data has no unseen "
            "speakers or reference-clip field."
        )

    wanted = set(speakers) if speakers is not None else None
    utts = [u for u in iter_split(root, split) if wanted is None or u.speaker_id in wanted]

    ref_by_speaker: Dict[str, List[str]] = {}
    if track == "A":
        for u in iter_split(root, reference_split):
            ref_by_speaker.setdefault(u.speaker_id, []).append(u.audio_rel)
        for k in ref_by_speaker:
            ref_by_speaker[k].sort()

    report = BuildReport()
    samples: List[Sample] = []
    seen_ids = set()
    no_ref = set()
    for u in sorted(utts, key=sample_id_for):
        sid = sample_id_for(u)
        if sid in seen_ids:
            raise DatasetFormatError(f"duplicate sample_id {sid}")
        seen_ids.add(sid)

        if track == "A":
            refs = [r for r in ref_by_speaker.get(u.speaker_id, []) if r != u.audio_rel]
            if max_reference_clips is not None:
                refs = refs[:max_reference_clips]
        else:
            refs = list(reference_map.get(sid, []))
        if not refs:
            no_ref.add(u.speaker_id)

        gen_rel: Optional[str] = None
        if source == "ground_truth":
            gen_rel = u.audio_rel
        else:
            for ext in generated_exts:
                if (Path(generated_dir) / f"{sid}{ext}").exists():
                    gen_rel = f"{sid}{ext}"
                    break
            if gen_rel is None:
                report.n_missing_generated += 1

        if not (root / u.audio_rel).exists():
            report.n_missing_ground_truth += 1
        report.language_ids[u.language_id] += 1
        samples.append(
            Sample.from_text(
                sid, u.speaker_id, u.text, parser,
                generated_audio=gen_rel,
                reference_audio=refs,
                ground_truth_audio=u.audio_rel,
                meta={"language_id": u.language_id},
            )
        )

    report.n_samples = len(samples)
    report.speakers_without_reference = sorted(no_ref)
    header = ManifestHeader(
        track=track,
        source=source,
        split=split,
        audio_root=Path(root).as_posix(),
        generated_root=Path(generated_dir).as_posix() if generated_dir is not None else (
            Path(root).as_posix() if source == "ground_truth" else None),
        nv_types=list(parser.nv_types),
    )
    manifest = Manifest(header, samples)
    manifest.validate(parser)
    return manifest, report


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Build an evaluation manifest from the released ViNV-TTS data.")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--track", default="A", choices=["A", "B"])
    ap.add_argument("--source", default="ground_truth", choices=["ground_truth", "model"])
    ap.add_argument("--generated-dir", type=Path)
    ap.add_argument("--max-reference-clips", type=int)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    m, rep = build_manifest(
        a.root, a.split, track=a.track, source=a.source, generated_dir=a.generated_dir,
        max_reference_clips=a.max_reference_clips,
    )
    m.save(a.out)
    print(f"wrote {a.out}\n{rep}")
    if rep.speakers_without_reference:
        print(f"WARNING speakers without reference audio: {rep.speakers_without_reference}")


if __name__ == "__main__":
    main()
