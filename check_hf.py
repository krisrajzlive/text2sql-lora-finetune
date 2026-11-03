"""Check the Hugging Face token setup without printing the token.

Usage: python check_hf.py
"""
import os

from huggingface_hub import whoami

from publish import load_env

NAMES = ("HF_TOKEN", "HUGGINGFACE_API_KEY")

if not os.path.exists(".env"):
    print(".env: not found in", os.getcwd())
else:
    print(".env found. Lines:")
    for n, line in enumerate(open(".env", encoding="utf-8-sig"), 1):
        line = line.rstrip("\r\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            stray = any(c in value for c in " \"'")
            print(f"  {n}: key={key!r} value_len={len(value)} stray_quote_or_space={stray}")
        else:
            print(f"  {n}: no '=' sign (line length {len(line)}); expected KEY=value")

load_env()
token = next((os.environ[n] for n in NAMES if os.environ.get(n)), "")
print("token loaded:", bool(token), "| length:", len(token), "| starts with hf_:", token.startswith("hf_"))
if token:
    try:
        info = whoami(token=token)
        role = info.get("auth", {}).get("accessToken", {}).get("role")
        print("HF account:", info["name"], "| token role:", role)
    except Exception as exc:
        print("HF rejected the token:", type(exc).__name__, str(exc)[:100])
