"""Command-line client.

Usage: python client.py "question" --schema "CREATE TABLE ..." [--url http://localhost:8000]
"""
import argparse
import json
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--schema", required=True)
ap.add_argument("--url", default="http://localhost:8000")
args = ap.parse_args()

req = urllib.request.Request(
    f"{args.url}/generate",
    data=json.dumps({"question": args.question, "schema": args.schema}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as resp:
    print(json.load(resp)["sql"])
