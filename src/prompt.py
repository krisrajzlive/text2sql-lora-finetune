"""Prompt template shared by training, evaluation and serving."""

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
