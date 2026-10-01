"""The one canonical parser for inline non-verbal (NV) tags.

Every other module consumes `ParsedText`; no other code may parse raw tags.

Conventions
-----------
* A tag is `[name]`. Names are matched case-insensitively after stripping
  whitespace, and stored WITHOUT brackets (e.g. "laughter").  [ASSUMPTION]
* Words are whitespace-delimited tokens of the tag-free text. For Vietnamese
  these are syllables.
* `gap_index` = number of words that precede the tag. 0 means before the first
  word, `n_words` means after the last word. Several NVs may share one gap;
  their order in `events` is the order in the text.
* A tag is always a token boundary: "a[laughter]b" yields words "a", "b". The
  released data never contains glued tags (verified), so this is only a
  defensive rule.
* `clean_text` = words joined by single spaces. No other normalisation is
  applied here (case, punctuation, Unicode form are left untouched); text
  normalisation for WER belongs to the WER module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

DEFAULT_NV_TYPES: Tuple[str, ...] = ("laughter", "breathing", "sniff", "throatclearing")
UNKNOWN_POLICIES = ("error", "drop", "keep")

_TAG_RE = re.compile(r"\[([^\[\]]*)\]")


class NVParseError(ValueError):
    """Raised for malformed transcripts (unbalanced brackets, unknown tag under 'error')."""


@dataclass(frozen=True)
class NVEvent:
    type: str
    gap_index: int

    def to_dict(self) -> Dict[str, object]:
        return {"type": self.type, "gap_index": self.gap_index}


@dataclass(frozen=True)
class ParsedText:
    raw: str
    clean_text: str
    words: Tuple[str, ...]
    events: Tuple[NVEvent, ...]
    unknown_tags: Tuple[str, ...] = ()  # raw unknown tags seen (only with policy drop/keep)

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def event_dicts(self) -> List[Dict[str, object]]:
        return [e.to_dict() for e in self.events]

    def tagged_text(self) -> str:
        """Re-render the canonical tagged form: single spaces, tags at their gaps."""
        out: List[str] = []
        ei = 0
        for i in range(len(self.words) + 1):
            while ei < len(self.events) and self.events[ei].gap_index == i:
                out.append(f"[{self.events[ei].type}]")
                ei += 1
            if i < len(self.words):
                out.append(self.words[i])
        return " ".join(out)


class NVParser:
    def __init__(self, nv_types: Iterable[str] = DEFAULT_NV_TYPES, unknown_policy: str = "error"):
        if unknown_policy not in UNKNOWN_POLICIES:
            raise ValueError(f"unknown_policy must be one of {UNKNOWN_POLICIES}")
        self.nv_types = tuple(t.strip().lower() for t in nv_types)
        self._known = frozenset(self.nv_types)
        self.unknown_policy = unknown_policy

    def parse(self, text: str) -> ParsedText:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")

        words: List[str] = []
        events: List[NVEvent] = []
        unknown: List[str] = []
        pos = 0
        for m in _TAG_RE.finditer(text):
            words.extend(text[pos : m.start()].split())
            pos = m.end()
            name = m.group(1).strip().lower()
            if name in self._known:
                events.append(NVEvent(name, len(words)))
                continue
            unknown.append(m.group(0))
            if self.unknown_policy == "error":
                raise NVParseError(f"unknown tag {m.group(0)!r} in: {text!r}")
            if self.unknown_policy == "keep":
                events.append(NVEvent(name, len(words)))
            # "drop": the tag disappears but still acts as a token boundary
        words.extend(text[pos:].split())

        if "[" in _TAG_RE.sub("", text) or "]" in _TAG_RE.sub("", text):
            raise NVParseError(f"unbalanced or nested brackets in: {text!r}")

        return ParsedText(
            raw=text,
            clean_text=" ".join(words),
            words=tuple(words),
            events=tuple(events),
            unknown_tags=tuple(unknown),
        )

    def parse_many(self, texts: Iterable[str]) -> List[ParsedText]:
        return [self.parse(t) for t in texts]
