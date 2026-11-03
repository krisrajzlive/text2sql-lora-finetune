# text2sql-lora-finetune

Parameter-efficient (LoRA) fine-tuning of `Qwen2.5-1.5B-Instruct` for text-to-SQL on a single 16 GB Tesla T4, with a before/after evaluation.

- **Data:** [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context): question + table schema -> SQL.
- **Method:** LoRA (r=16) on all attention and MLP projections; fp16 base weights, fp32 adapters, gradient checkpointing. Loss is computed on answer tokens only.
- **Eval:** exact match (normalised whitespace, case, trailing `;`) on 200 held-out examples, base vs fine-tuned.
- **T4 notes:** no bf16 or FlashAttention-2, so training uses fp16 AMP. Checkpoints every 100 steps and training resumes automatically if the session is interrupted.

## Run

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

git clone <this-repo-url> && cd text2sql-lora-finetune
pip install -r requirements.txt

python train.py --config configs/qwen25_1p5b.yaml
python eval.py  --config configs/qwen25_1p5b.yaml
```

Results are written to `outputs/eval_results.json`; the adapter is saved to `outputs/qwen25-1p5b-text2sql/adapter`.

If you hit out-of-memory errors, lower `per_device_batch_size` (raise `grad_accum_steps` to compensate) or `max_length` in the config. To try a smaller model, set `model_name: Qwen/Qwen2.5-0.5B-Instruct`.

## Layout

```
configs/   training hyperparameters
src/data.py   dataset loading, prompt template, label masking
train.py   LoRA training with checkpoint resume
eval.py    base vs fine-tuned exact-match comparison
```
