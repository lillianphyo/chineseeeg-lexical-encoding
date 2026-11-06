"""Character surprisal for every ChineseEEG (ds004952) run, from a character-level Chinese GPT-2 (text only, no EEG).

Plan: lab01/.claude/prds/exp-chineseeeg-overlap-regression-2026-10.md (Gate 0: model fixed before any EEG fit).
Context = the preceding text of the same run, windows of 512 characters stepping by 256.
Writes surprisal.npz: per run id, arrays rows / positions / bits of the highlighted characters.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from zuco_llm_align.chineseeeg import RUN_ORDER, run_id, units_of
from zuco_llm_align.chineseeeg_regression import highlighted_surprisal

MODEL = "uer/gpt2-chinese-cluecorpussmall"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", default=os.environ.get("CHINESEEEG_ROOT", str(Path.home() / "datasets/ds004952")))
    p.add_argument("--out", default=str(Path.home() / "datasets/ds004952_chartable/surprisal.npz"))
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()

    @torch.no_grad()
    def logprobs(ids):
        return model(torch.as_tensor(ids)[None]).logits[0].log_softmax(-1).numpy()

    arrays = {}
    for session, run in RUN_ORDER:
        table = highlighted_surprisal(units_of(args.root, session, run), tok.vocab.get, logprobs, tok.unk_token_id)
        keys = sorted(table)
        rid = run_id(session, run)
        arrays[f"{rid}_rows"] = np.array([r for r, _ in keys])
        arrays[f"{rid}_positions"] = np.array([k for _, k in keys])
        arrays[f"{rid}_bits"] = np.array([table[k] for k in keys])
        bits = arrays[f"{rid}_bits"]
        print(f"{session} run {run}: {len(bits)} characters, mean {np.nanmean(bits):.2f} bits, {np.isnan(bits).sum()} undefined", flush=True)
    np.savez(args.out, model=MODEL, **arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
