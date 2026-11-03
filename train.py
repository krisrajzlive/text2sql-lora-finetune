"""LoRA fine-tuning for text-to-SQL.  Usage: python train.py --config configs/qwen25_1p5b.yaml"""
import argparse
import os

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from src.data import load_splits, tokenize_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc, lc = cfg["train"], cfg["lora"]

    assert torch.cuda.is_available(), "No GPU visible; run nvidia-smi."
    set_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # T4 has no bf16: keep frozen base in fp16, LoRA weights train in fp32 (peft default).
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    train_ds, _ = load_splits(cfg)
    train_ds = train_ds.map(
        lambda ex: tokenize_example(tok, ex, cfg["max_length"]),
        remove_columns=train_ds.column_names,
    )

    steps_per_epoch = -(-len(train_ds) // (tc["per_device_batch_size"] * tc["grad_accum_steps"]))
    warmup_steps = max(1, int(tc["warmup_ratio"] * steps_per_epoch * tc["epochs"]))

    args_tr = TrainingArguments(
        output_dir=tc["output_dir"],
        num_train_epochs=tc["epochs"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum_steps"],
        learning_rate=tc["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=tc["logging_steps"],
        save_steps=tc["save_steps"],
        save_total_limit=tc["save_total_limit"],
        report_to="none",
        seed=cfg["seed"],
    )
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    last = get_last_checkpoint(tc["output_dir"]) if os.path.isdir(tc["output_dir"]) else None
    trainer.train(resume_from_checkpoint=last)

    final = os.path.join(tc["output_dir"], "adapter")
    model.save_pretrained(final)  # adapter only (~tens of MB)
    tok.save_pretrained(final)
    print(f"Saved adapter to {final}")


if __name__ == "__main__":
    main()
