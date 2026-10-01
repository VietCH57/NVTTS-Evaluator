"""Canonical evaluation manifest.

Everything downstream of the dataset adapter works only with this structure.

File format: JSON Lines. Line 1 is `{"_header": {...}}`, each following line is one
sample. Paths are stored with forward slashes and are RELATIVE:
  * ground_truth_audio, reference_audio -> relative to header.audio_root
  * generated_audio                     -> relative to header.generated_root
(absolute paths are passed through unchanged).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .nv_parser import DEFAULT_NV_TYPES, NVParser

SCHEMA_VERSION = 1
TRACKS = ("A", "B")


class ManifestError(ValueError):
    pass


@dataclass
class Sample:
    sample_id: str
    speaker_id: str
    text: str                      # raw transcript with inline tags
    clean_text: str                # produced by the canonical parser
    nv_events: List[Dict[str, Any]]  # [{"type": str, "gap_index": int}, ...]
    generated_audio: Optional[str] = None
    reference_audio: List[str] = field(default_factory=list)
    ground_truth_audio: Optional[str] = None
    duration: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(cls, sample_id: str, speaker_id: str, text: str, parser: NVParser, **kwargs) -> "Sample":
        parsed = parser.parse(text)
        return cls(
            sample_id=sample_id,
            speaker_id=speaker_id,
            text=text,
            clean_text=parsed.clean_text,
            nv_events=parsed.event_dicts,
            **kwargs,
        )


@dataclass
class ManifestHeader:
    track: str
    source: str = "unspecified"        # e.g. "ground_truth" or "model"
    split: Optional[str] = None
    audio_root: Optional[str] = None
    generated_root: Optional[str] = None
    nv_types: List[str] = field(default_factory=lambda: list(DEFAULT_NV_TYPES))
    schema_version: int = SCHEMA_VERSION


class Manifest:
    def __init__(self, header: ManifestHeader, samples: List[Sample]):
        self.header = header
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    # ---- paths -----------------------------------------------------------------
    @staticmethod
    def _join(root: Optional[str], rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute() or root is None:
            return p
        return Path(root) / p

    def resolve_ground_truth(self, s: Sample) -> Optional[Path]:
        return self._join(self.header.audio_root, s.ground_truth_audio) if s.ground_truth_audio else None

    def resolve_references(self, s: Sample) -> List[Path]:
        return [self._join(self.header.audio_root, r) for r in s.reference_audio]

    def resolve_generated(self, s: Sample) -> Optional[Path]:
        return self._join(self.header.generated_root, s.generated_audio) if s.generated_audio else None

    # ---- validation ------------------------------------------------------------
    def validate(self, parser: Optional[NVParser] = None) -> None:
        """Raise ManifestError listing problems. Re-parses every transcript so the
        stored clean_text / nv_events can never drift from the canonical parser."""
        parser = parser or NVParser(self.header.nv_types)
        problems: List[str] = []
        if self.header.track not in TRACKS:
            problems.append(f"track must be one of {TRACKS}, got {self.header.track!r}")
        seen = set()
        for s in self.samples:
            if s.sample_id in seen:
                problems.append(f"duplicate sample_id: {s.sample_id}")
            seen.add(s.sample_id)
            try:
                p = parser.parse(s.text)
            except ValueError as e:
                problems.append(f"{s.sample_id}: {e}")
                continue
            if p.clean_text != s.clean_text:
                problems.append(f"{s.sample_id}: clean_text differs from parser output")
            if p.event_dicts != s.nv_events:
                problems.append(f"{s.sample_id}: nv_events differ from parser output")
            if self.header.track == "B" and not s.reference_audio:
                problems.append(f"{s.sample_id}: Track B requires reference_audio")
        if problems:
            shown = "\n  ".join(problems[:20])
            more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
            raise ManifestError(f"{len(problems)} manifest problem(s):\n  {shown}{more}")

    # ---- io --------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"_header": asdict(self.header)}, ensure_ascii=False) + "\n")
            for s in self.samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path, parser: Optional[NVParser] = None, validate: bool = True) -> "Manifest":
        with open(Path(path), "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if not lines or "_header" not in json.loads(lines[0]):
            raise ManifestError("first line must be a {'_header': ...} record")
        header = ManifestHeader(**json.loads(lines[0])["_header"])
        if header.schema_version != SCHEMA_VERSION:
            raise ManifestError(f"unsupported schema_version {header.schema_version}")
        samples = [Sample(**json.loads(ln)) for ln in lines[1:]]
        m = cls(header, samples)
        if validate:
            m.validate(parser)
        return m

    def subset(self, sample_ids: Iterable[str]) -> "Manifest":
        wanted = set(sample_ids)
        return Manifest(self.header, [s for s in self.samples if s.sample_id in wanted])
