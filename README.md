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

## 2. Publish the adapter

Create a Hugging Face access token with write permission and keep it out of the repo.

```bash
export HF_TOKEN=<your token>
python publish.py --config configs/qwen25_1p5b.yaml --repo-id <hf-user>/text2sql-qwen25-1p5b-lora
```

This uploads the adapter and a model card containing the eval numbers (use `--private` for a private repo).

## 3. Serve

Any machine with Docker works; no GPU is required (CPU is slower, roughly seconds per query).

```bash
export ADAPTER_ID=<hf-user>/text2sql-qwen25-1p5b-lora
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
train.py  eval.py  publish.py
serve/api.py    FastAPI service
ui/app.py       Streamlit front end
client.py       CLI client
Dockerfile  docker-compose.yml
```
