"""Publish the trained LoRA adapter to the Hugging Face Hub.

Usage: HF_TOKEN=... python publish.py --config configs/qwen25_1p5b.yaml --repo-id <user>/<name>
"""
import argparse
import json
import os
import tempfile

import yaml
from huggingface_hub import HfApi

CARD = """---
base_model: {base}
library_name: peft
license: apache-2.0
tags: [lora, text-to-sql, peft]
datasets: [{dataset}]
---

# {name}

LoRA adapter for `{base}` fine-tuned for text-to-SQL on `{dataset}`.

{metrics}## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{base}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{base}"), "{repo}")
```

The prompt template is defined in the training repository (`src/prompt.py`).
"""


def load_env(path=".env"):
    """Minimal .env loader: KEY=VALUE lines; existing environment variables win."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--results", default="outputs/eval_results.json")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    adapter = args.adapter or f"{cfg['train']['output_dir']}/adapter"

    metrics = ""
    if os.path.exists(args.results):
        r = json.load(open(args.results))
        metrics = (f"## Evaluation\n\nExact match on {r['n']} held-out examples: "
                   f"base **{r['base_em']:.3f}**, fine-tuned **{r['tuned_em']:.3f}**.\n\n")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY")
    api = HfApi(token=token)  # falls back to a cached login when None
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=adapter)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "README.md")
        with open(path, "w") as f:
            f.write(CARD.format(
                base=cfg["model_name"], dataset=cfg["dataset"], name=args.repo_id.split("/")[-1],
                repo=args.repo_id, metrics=metrics))
        api.upload_file(path_or_fileobj=path, path_in_repo="README.md", repo_id=args.repo_id)
    print(f"Published https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
