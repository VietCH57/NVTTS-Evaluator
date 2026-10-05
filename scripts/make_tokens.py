"""Generate tokens.txt for sherpa-onnx from the model's SentencePiece bpe.model.

    python scripts/make_tokens.py --bpe D:\\Work\\models\\Zipformer-30M-RNNT-6000h\\bpe.model ^
        --out D:\\Work\\models\\Zipformer-30M-RNNT-6000h\\tokens.txt

Sanity check afterwards: the number of lines must equal the model's vocabulary size, and the
first lines should look like "<blk> 0", "<sos/eos> 1", "<unk> 2" (icefall convention).
If a transcript looks like garbage, compare with the tokens.txt of sherpa-onnx's packaged
release of the same model.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nvtts_eval.models.tokens import write_tokens  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    n = write_tokens(a.bpe, a.out)
    print(f"wrote {a.out}: {n} tokens")
    print("".join(a.out.read_text(encoding="utf-8").splitlines(True)[:6]), end="")


if __name__ == "__main__":
    main()
