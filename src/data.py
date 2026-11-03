"""Dataset loading and prompt formatting for text-to-SQL."""
from datasets import load_dataset

SYSTEM = "You translate questions into SQL using the given schema. Reply with SQL only."


def build_messages(example):
    user = f"Schema:\n{example['context']}\n\nQuestion: {example['question']}"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]


def prompt_text(tokenizer, example):
    return tokenizer.apply_chat_template(
        build_messages(example), tokenize=False, add_generation_prompt=True
    )


def load_splits(cfg):
    ds = load_dataset(cfg["dataset"], split="train").shuffle(seed=cfg["seed"])
    n_eval, n_train = cfg["eval_samples"], cfg["train_samples"]
    return ds.select(range(n_eval, n_eval + n_train)), ds.select(range(n_eval))


def tokenize_example(tokenizer, example, max_length):
    """Tokenize prompt+answer; loss is computed on the answer tokens only."""
    prompt_ids = tokenizer(prompt_text(tokenizer, example), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["answer"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
