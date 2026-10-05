import numpy as np
import pytest
import soundfile as sf

from nvtts_eval.data import Manifest, ManifestHeader, NVParser, Sample


def write_wav(path, n_samples, sr=16000, value=0.1):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.full(n_samples, value, dtype="float32"), sr)


@pytest.fixture
def parser():
    return NVParser()


@pytest.fixture
def make_manifest(tmp_path):
    """Manifest whose generated/reference wavs are 16 kHz files; the length of each file
    (in seconds) is its 'identity' for the fake backends below."""
    def _make(items, track="A"):
        # items: list of (sample_id, speaker, text, gen_seconds, [ref_seconds,...])
        samples = []
        for sid, spk, text, gen_s, refs in items:
            write_wav(tmp_path / "gen" / f"{sid}.wav", int(16000 * gen_s))
            ref_names = []
            for k, rs in enumerate(refs):
                name = f"{spk}_ref{k}_{int(rs*1000)}.wav"
                write_wav(tmp_path / "ref" / name, int(16000 * rs))
                ref_names.append(f"ref/{name}")
            samples.append(Sample.from_text(sid, spk, text, NVParser(), generated_audio=f"{sid}.wav",
                                            reference_audio=ref_names))
        hdr = ManifestHeader(track=track, source="model", split="dev",
                             audio_root=tmp_path.as_posix(), generated_root=(tmp_path / "gen").as_posix())
        return Manifest(hdr, samples)
    return _make
