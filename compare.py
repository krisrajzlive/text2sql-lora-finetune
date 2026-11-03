"""Show base-model output next to fine-tuned output for the same prompts.

Examples:
  python compare.py                                   # built-in example questions
  python compare.py "Top 3 salaries" --schema "CREATE TABLE employees (id INT, salary INT)"
  python compare.py --heldout 5                       # held-out dataset rows, with gold SQL (needs `datasets`)

One model load serves both columns: the adapter is switched off for the "base" output.
"""
import argparse
import os

DEFAULT_ADAPTER = "krisrajz/text2sql-qwen25-1p5b-lora"
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"

EXAMPLES = [
    ("How many employees are in the sales department?",
     "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)"),
    ("What is the average salary of employees earning more than 50000?",
     "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary INT)"),
    ("List the names of customers who placed an order in 2023.",
     "CREATE TABLE customers (id INT, name VARCHAR); CREATE TABLE orders (id INT, customer_id INT, order_year INT)"),
]


def heldout_examples(n, config_path):
    import yaml
    from src.data import load_splits

    cfg = yaml.safe_load(open(config_path))
    _, eval_ds = load_splits(cfg)
    return [(eval_ds[i]["question"], eval_ds[i]["context"], eval_ds[i]["answer"]) for i in range(n)]


def show(title, text):
    print(f"  {title:<11}: " + text.replace("\n", "\n" + " " * 15))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--schema")
    ap.add_argument("--heldout", type=int, metavar="N", help="use N held-out dataset rows (shows gold SQL)")
    ap.add_argument("--config", default="configs/qwen25_1p5b.yaml")
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER_ID", DEFAULT_ADAPTER))
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE))
    args = ap.parse_args()

    if args.question and not args.schema:
        ap.error("--schema is required with a question")
    if args.question:
        items = [(args.question, args.schema, None)]
    elif args.heldout:
        items = heldout_examples(args.heldout, args.config)
    else:
        items = [(q, s, None) for q, s in EXAMPLES]

    from src.inference import Text2SQL  # heavy import deferred so --help and arg errors are instant

    print(f"Loading {args.base_model} + {args.adapter} ...")
    model = Text2SQL(args.base_model, args.adapter)
    print(f"Device: {model.device}\n")

    for i, (question, schema, gold) in enumerate(items, 1):
        print(f"[{i}] {question}")
        print(f"  schema     : {schema}")
        if gold:
            show("gold", gold)
        show("base", model.generate(question, schema, use_adapter=False))
        show("fine-tuned", model.generate(question, schema, use_adapter=True))
        print()


if __name__ == "__main__":
    main()
