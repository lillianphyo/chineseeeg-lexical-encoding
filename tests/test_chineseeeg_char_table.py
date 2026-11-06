import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("chineseeeg_char_table", Path(__file__).parents[1] / "scripts" / "chineseeeg_char_table.py")
char_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(char_table)


def test_unihan_strokes_take_the_first_count():
    lines = ["# comment", "U+4E00\tkTotalStrokes\t1", "U+4E8C\tkTotalStrokes\t2 3", "U+4E00\tkIRG_GSource\tG0-523B"]
    assert char_table.unihan_strokes(lines) == {"一": 1, "二": 2}


def test_build_drops_characters_the_tokenizer_maps_to_unk():
    vocab = {"一": 1, "二": 2}
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
    chars, vectors, strokes, dropped = char_table.build(
        ["二", "—", "一"], lambda c: [c.lower()], lambda toks: [vocab.get(t, 0) for t in toks], matrix, 0, {"一": 1})
    assert chars == ["二", "一"] and dropped == ["—"]
    assert vectors.tolist() == [matrix[2].tolist(), matrix[1].tolist()]
    assert strokes.tolist() == [-1, 1]
