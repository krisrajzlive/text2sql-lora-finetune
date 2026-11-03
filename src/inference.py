"""Load the base model plus the LoRA adapter and generate SQL."""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompt import prompt_text


class Text2SQL:
    def __init__(self, base_model, adapter, device=None):
        """`adapter` is a local directory or a Hugging Face Hub repo id."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        self.model = PeftModel.from_pretrained(base, adapter).to(self.device).eval()

    @torch.no_grad()
    def generate(self, question, schema, max_new_tokens=128, use_adapter=True):
        """With use_adapter=False the LoRA weights are bypassed, giving the plain base model."""
        prompt = prompt_text(self.tok, {"question": question, "context": schema})
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        if use_adapter:
            out = self.model.generate(**enc, **kwargs)
        else:
            with self.model.disable_adapter():
                out = self.model.generate(**enc, **kwargs)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
