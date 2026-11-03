# text2sql-lora-finetune

End-to-end lifecycle for a LoRA fine-tuned text-to-SQL model: **train, evaluate, publish, serve, use.**

- **Model:** `Qwen2.5-1.5B-Instruct` + LoRA adapter (r=16, all attention and MLP projections).
- **Data:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context): question + table schema -> SQL.
- **Training hardware:** a single 16 GB Tesla T4 (fp16, no bf16 or FlashAttention-2). Checkpoints every 100 steps and training resumes automatically.
- **Eval:** exact match (normalised whitespace, case, trailing `;`) on 200 held-out examples, base vs fine-tuned.

```
Train (GPU)  ->  Eval  ->  Publish adapter (HF Hub)  ->  Serve (FastAPI, Docker)  ->  Use (Streamlit UI / CLI / REST)
```

## 1. Train and evaluate (GPU workspace)

```bash
pip install -r requirements.txt
python train.py --config configs/qwen25_1p5b.yaml
python eval.py  --config configs/qwen25_1p5b.yaml
```

Output: adapter in `outputs/qwen25-1p5b-text2sql/adapter`, metrics in `outputs/eval_results.json`.
On out-of-memory errors lower `per_device_batch_size` (raise `grad_accum_steps` to compensate) in the config.

**Notes for the GPU workspace**

- Confirm the GPU with `nvidia-smi`, then confirm PyTorch sees it: `python -c "import torch; print(torch.cuda.is_available())"`.
  If the preinstalled PyTorch is a CPU build (`+cpu`), install a CUDA build first:
  `pip install --user --force-reinstall torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128`.
- The code targets transformers 5.x (`dtype=`, `warmup_steps`; `group_by_length` is not used). `pip install -r requirements.txt` is enough on a fresh machine.
- Training 6,000 examples for one epoch (375 steps) takes about 11 minutes on a T4.

### Results

| Model | Exact match (n=200 held-out) |
|---|---|
| Base `Qwen2.5-1.5B-Instruct` | 1.5% |
| + LoRA fine-tune | 71.0% |

The base score understates the base model: it tends to wrap SQL in explanations or markdown, so a strict string match
mostly measures output format. The 71.0% is also a strict match, so semantically equivalent queries (different aliases or column order)
count as misses. Execution-based accuracy would be a fairer metric.

## 2. Publish the adapter

Publishing needs a Hugging Face access token that can write to repos. Create one at
<https://huggingface.co/settings/tokens>: either a classic **Write** token, or a fine-grained token with
**write access to repos** under your own account. Do not paste it into chat, issues or commits.

Store it in a git-ignored `.env` file (see `.env.example`):

```bash
cp .env.example .env     # then set the token; .env is listed in .gitignore
```

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

`HUGGINGFACE_API_KEY` is accepted as an alias for `HF_TOKEN`. The key name must match exactly, and the line must
have no spaces around `=` and no quotes. `publish.py` and `docker compose` read the file automatically; an exported
shell variable of the same name takes priority over the file. `.env` is a dotfile, so `ls` hides it
(use `ls -a`) and some editors hide it from the file explorer.

Check the setup before publishing (prints key names and lengths, never the token):

```bash
python check_hf.py
```

Expected output: the key name, `token loaded: True`, and `HF account: <your-username> | token role: ...`.
Use that exact username in `--repo-id`. Common failures:

| Symptom | Cause |
|---|---|
| `token loaded: False` | Misspelled key name, or `.env` is not in the current folder |
| `Invalid user token` / `401 Unauthorized` | Token revoked, copied incompletely, or wrong |
| `403 Forbidden` | Token has no write permission, or `--repo-id` uses a username that is not yours |

```bash
python publish.py --config configs/qwen25_1p5b.yaml --repo-id <hf-user>/text2sql-qwen25-1p5b-lora
```

This uploads the adapter and a model card containing the eval numbers (use `--private` for a private repo).
Publishing needs only `outputs/` and the repo, so it also works from a laptop: download `outputs/` from the GPU
workspace, then run the command there.

The GPU workspace is temporary. After publishing, delete `.env` and the Hugging Face cache (`~/.cache/huggingface`) from it.

## 3. Serve

Any machine with Docker works; no GPU is required (CPU is slower, roughly seconds per query).

Set `ADAPTER_ID` in `.env` (or export it), then:

```bash
docker compose up --build
```

The first start downloads the base model into a Docker volume. Endpoints:

- API: `http://localhost:8000` (`GET /health`, `POST /generate`, docs at `/docs`)
- UI: `http://localhost:8501`

For a GPU host build the image with `--build-arg BASE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime --build-arg INSTALL_TORCH=0`.

## 4. Use

```bash
python client.py "How many employees are in sales?" \
  --schema "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR)"
```

```bash
curl -s localhost:8000/generate -H 'Content-Type: application/json' \
  -d '{"question": "How many employees are in sales?", "schema": "CREATE TABLE employees (id INT, department VARCHAR)"}'
```

## Tests

```bash
pip install fastapi httpx pytest
python -m pytest tests
```

The API tests use a stub model, so they run without a GPU or model download.

## Layout

```
configs/        training hyperparameters
src/prompt.py   prompt template shared by train, eval and serving
src/data.py     dataset loading and label masking
src/inference.py  base model + adapter loader
train.py  eval.py  publish.py  check_hf.py
serve/api.py    FastAPI service
ui/app.py       Streamlit front end
client.py       CLI client
Dockerfile  docker-compose.yml
```
