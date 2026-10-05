"""Evaluation configuration (YAML or JSON). Unknown keys are errors (catches typos).

Everything here marked [ASSUMPTION] is a development choice, not an official rule,
and is echoed in every summary.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .scoring.formulas import ScoringConfig
from .metrics.wer import TextNormConfig


@dataclass
class AsrConfig:
    backend: str = "sherpa_onnx_offline"
    model_dir: Optional[str] = None       # folder with encoder/decoder/joiner .onnx + tokens.txt
    use_int8: bool = False                # [ASSUMPTION] fp32 by default
    num_threads: int = 2
    provider: str = "cpu"
    decoding_method: str = "greedy_search"


@dataclass
class PMosConfig:
    repo_id: str = "prj-beatrice/dnsmos-torch-native"
    revision: Optional[str] = None        # pin a commit hash (the repo ships remote code)
    device: str = "auto"
    output: str = "ovrl"                  # [ASSUMPTION] which DNSMOS output is "pMOS": sig|bak|ovrl|p808


@dataclass
class SSConfig:
    source: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: Optional[str] = None
    device: str = "auto"
    ref_mode: str = "centroid"            # [ASSUMPTION] centroid | mean_cosine
    max_reference_clips: Optional[int] = None


@dataclass
class WerConfig:
    aggregation: str = "corpus"           # [ASSUMPTION] corpus (sum errors / sum words) | sample_mean


@dataclass
class BootstrapConfig:
    n_boot: int = 2000
    alpha: float = 0.05
    seed: int = 0
    min_n: int = 10                       # groups with fewer valid values are flagged low_n


@dataclass
class NvpaConfig:
    detector_path: Optional[str] = None   # trained with `train-detector`
    tolerance_words: int = 1              # [ASSUMPTION] position tolerance in reference words
    min_window: float = 0.2
    max_window: float = 3.0
    unreliable_policy: str = "miss"       # [ASSUMPTION] events that cannot be localised: miss | exclude
    shuffle_reps: int = 50


@dataclass
class HumanConfig:
    subset_size: int = 100
    seed: int = 0
    rare_type_share: float = 0.05         # NV types with <= this share of events are "rare" and over-sampled
    max_rare_fraction: float = 0.5        # at most this fraction of the subset is reserved for rare types
    long_utt_seconds: float = 15.0        # stratification: short vs long utterances
    head_speaker_min_utts: int = 20       # stratification: head vs tail speakers
    n_gt_anchors: int = 5                 # hidden ground-truth items per rating package
    n_degraded_anchors: int = 5           # hidden artificially degraded items
    degrade_snr_db: float = 5.0
    anchor_gt_min: float = 3.0            # rater flagged if mean SN on ground-truth anchors is below this
    anchor_gap_min: float = 0.5           # ... or if ground-truth anchors are not rated this much above degraded ones
    exclude_flagged: bool = True
    auto_on_subset: bool = True           # [ASSUMPTION] final score uses automatic metrics on the rated subset


@dataclass
class EvalConfig:
    asr: AsrConfig = field(default_factory=AsrConfig)
    pmos: PMosConfig = field(default_factory=PMosConfig)
    ss: SSConfig = field(default_factory=SSConfig)
    wer: WerConfig = field(default_factory=WerConfig)
    text_norm: TextNormConfig = field(default_factory=TextNormConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    nvpa: NvpaConfig = field(default_factory=NvpaConfig)
    human: HumanConfig = field(default_factory=HumanConfig)

    def validate(self) -> None:
        if self.asr.backend != "sherpa_onnx_offline":
            raise ValueError(f"unsupported asr.backend {self.asr.backend!r}")
        if self.pmos.output not in ("sig", "bak", "ovrl", "p808"):
            raise ValueError("pmos.output must be one of sig|bak|ovrl|p808")
        if self.ss.ref_mode not in ("centroid", "mean_cosine"):
            raise ValueError("ss.ref_mode must be centroid|mean_cosine")
        if self.wer.aggregation not in ("corpus", "sample_mean"):
            raise ValueError("wer.aggregation must be corpus|sample_mean")
        if self.nvpa.unreliable_policy not in ("miss", "exclude"):
            raise ValueError("nvpa.unreliable_policy must be miss|exclude")
        if self.nvpa.tolerance_words < 0:
            raise ValueError("nvpa.tolerance_words must be >= 0")

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: Dict[str, Any], where: str):
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(default):
            v = _build(type(default), v or {}, f"{where}.{f.name}" if where else f.name)
        kwargs[f.name] = v
    return cls(**kwargs)


def config_from_dict(data: Optional[Dict[str, Any]]) -> EvalConfig:
    cfg = _build(EvalConfig, data or {}, "")
    cfg.validate()
    return cfg


def load_config(path: Optional[Path]) -> EvalConfig:
    if path is None:
        return config_from_dict({})
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("PyYAML is required for YAML configs: pip install pyyaml") from e
        data = yaml.safe_load(text)
    return config_from_dict(data)
