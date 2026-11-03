"""Compare base vs fine-tuned model on held-out examples.

Usage: python eval.py --config configs/qwen25_1p5b.yaml [--adapter outputs/.../adapter]
Metric: exact match after whitespace/case/semicolon normalisation.
"""
import argparse
import json
import re

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import load_splits, prompt_text


def normalize(sql):
    sql = sql.strip().strip("`").replace("sql\n", "", 1) if sql.strip().startswith("```") else sql
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


@torch.no_grad()
def generate(model, tok, examples, batch_size=16):
    outs = []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [prompt_text(tok, ex) for ex in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


def score(preds, examples):
    hits = sum(normalize(p) == normalize(e["answer"]) for p, e in zip(preds, examples))
    return hits / len(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="outputs/eval_results.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    adapter = args.adapter or f"{cfg['train']['output_dir']}/adapter"

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _, eval_ds = load_splits(cfg)
    examples = [eval_ds[i] for i in range(len(eval_ds))]

    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float16).cuda().eval()
    base_preds = generate(base, tok, examples)
    base_em = score(base_preds, examples)

    tuned = PeftModel.from_pretrained(base, adapter).eval()
    tuned_preds = generate(tuned, tok, examples)
    tuned_em = score(tuned_preds, examples)

    print(f"Exact match  base: {base_em:.3f}   fine-tuned: {tuned_em:.3f}   (n={len(examples)})")
    samples = [
        {"question": e["question"], "gold": e["answer"], "base": b, "tuned": t}
        for e, b, t in list(zip(examples, base_preds, tuned_preds))[:10]
    ]
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"base_em": base_em, "tuned_em": tuned_em, "n": len(examples), "samples": samples},
              open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
