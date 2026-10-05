"""Word-level edit alignment shared by WER and NVPA.

`align` returns the pairs of an optimal (minimum edit distance) alignment, in order:
    (i, j)     reference word i aligned to hypothesis word j (match or substitution)
    (i, None)  reference word i deleted
    (None, j)  hypothesis word j inserted
Ties are broken deterministically: diagonal first, then deletion, then insertion.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Pair = Tuple[Optional[int], Optional[int]]


def align(ref: Sequence[str], hyp: Sequence[str]) -> List[Pair]:
    n, m = len(ref), len(hyp)
    table = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev, cur = table[-1], [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j - 1] + cost, prev[j] + 1, cur[j - 1] + 1)
        table.append(cur)
    i, j = n, m
    out: List[Pair] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and table[i][j] == table[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    out.reverse()
    return out


def counts(pairs: Sequence[Pair], ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int]:
    """(substitutions, deletions, insertions) of an alignment."""
    sub = sum(1 for i, j in pairs if i is not None and j is not None and ref[i] != hyp[j])
    dele = sum(1 for i, j in pairs if j is None)
    ins = sum(1 for i, j in pairs if i is None)
    return sub, dele, ins
