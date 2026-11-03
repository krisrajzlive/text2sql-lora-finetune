"""Generate SQL with the fine-tuned model directly (no API server needed).

Examples:
  python infer.py "How many employees are in sales?" --schema "CREATE TABLE employees (id INT, department VARCHAR)"
  python infer.py "Top 3 salaries" --schema-file schema.sql
  python infer.py --interactive --schema-file schema.sql

The adapter is read from --adapter, then $ADAPTER_ID, then the published default. It may be a Hugging Face repo id or a
local adapter folder (e.g. outputs/qwen25-1p5b-text2sql/adapter).
"""
import argparse
import os
import sys

from src.inference import Text2SQL

DEFAULT_ADAPTER = "krisrajz/text2sql-qwen25-1p5b-lora"
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="natural-language question")
    ap.add_argument("--schema", help="CREATE TABLE statement(s)")
    ap.add_argument("--schema-file", help="file containing the schema")
    ap.add_argument("--interactive", action="store_true", help="prompt for questions in a loop")
    ap.add_argument("--adapter", default=os.environ.get("ADAPTER_ID", DEFAULT_ADAPTER))
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE))
    args = ap.parse_args()

    schema = args.schema
    if args.schema_file:
        with open(args.schema_file, encoding="utf-8") as f:
            schema = f.read()
    if not schema:
        ap.error("provide --schema or --schema-file")
    if not args.question and not args.interactive:
        ap.error("provide a question or --interactive")

    print(f"Loading {args.base_model} + {args.adapter} ...", file=sys.stderr)
    model = Text2SQL(args.base_model, args.adapter)
    print(f"Device: {model.device}", file=sys.stderr)

    if args.question:
        print(model.generate(args.question, schema))

    if args.interactive:
        print("Enter a question (empty line to quit).", file=sys.stderr)
        while True:
            try:
                question = input("question> ").strip()
            except EOFError:
                break
            if not question:
                break
            print(model.generate(question, schema))


if __name__ == "__main__":
    main()
