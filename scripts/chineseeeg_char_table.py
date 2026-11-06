"""Character table for lab01/.claude/prds/exp-chineseeeg-char-decoding-2026-10.md: every highlighted character of
the novels -> static bert-base-chinese input token embedding (768d) and Unihan kTotalStrokes. Text only, no EEG.

    python scripts/chineseeeg_char_table.py --unihan ~/datasets/unihan/Unihan.zip --out ~/datasets/ds004952_chartable/char_table.npz
"""
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np

from zuco_llm_align.chineseeeg import RUN_ORDER, highlighted, units_of


def unihan_strokes(lines) -> dict[str, int]:
    out = {}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 3 and parts[1] == "kTotalStrokes":
            out[chr(int(parts[0][2:], 16))] = int(parts[2].split()[0])
    return out


def build(chars, tokenize, to_ids, matrix: np.ndarray, unk_id: int, strokes: dict[str, int]):
    """Characters with a single known token, their embedding rows and stroke counts, and the dropped characters."""
    kept, rows, dropped = [], [], []
    for c in chars:
        ids = to_ids(tokenize(c))
        if len(ids) == 1 and ids[0] != unk_id:
            kept.append(c)
            rows.append(ids[0])
        else:
            dropped.append(c)
    return kept, matrix[rows].astype(np.float32), np.array([strokes.get(c, -1) for c in kept]), dropped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.environ.get("CHINESEEEG_ROOT", str(Path.home() / "datasets/ds004952")))
    p.add_argument("--unihan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="bert-base-chinese")
    args = p.parse_args()
    from transformers import BertModel, BertTokenizer

    chars = sorted({c for s, r in RUN_ORDER for u in units_of(args.root, s, r) for c in highlighted(u)})
    with zipfile.ZipFile(args.unihan) as z:
        strokes = unihan_strokes(z.read("Unihan_IRGSources.txt").decode("utf-8").splitlines())
    tok = BertTokenizer.from_pretrained(args.model)
    matrix = BertModel.from_pretrained(args.model).embeddings.word_embeddings.weight.detach().numpy()
    kept, vectors, n_strokes, dropped = build(chars, tok.tokenize, tok.convert_tokens_to_ids, matrix, tok.unk_token_id, strokes)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, chars=np.array(kept), vectors=vectors, strokes=n_strokes)
    print(f"{len(kept)} characters kept, {len(dropped)} dropped {dropped}, {int((n_strokes < 0).sum())} without strokes")


if __name__ == "__main__":
    main()
