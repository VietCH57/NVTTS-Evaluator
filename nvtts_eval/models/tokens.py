"""Create sherpa-onnx's tokens.txt ("<piece> <id>" per line) from a SentencePiece bpe.model."""
from __future__ import annotations

from pathlib import Path
from typing import Union


def write_tokens(bpe_model: Union[str, Path], out_path: Union[str, Path]) -> int:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_model))
    n = sp.get_piece_size()
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            f.write(f"{sp.id_to_piece(i)} {i}\n")
    return n
