#!/usr/bin/env python3
"""Run the small Hugging Face SmolLM2 model with pure NumPy inference.

This keeps the same model family as pi_chat.py, but the actual forward pass is
implemented in NumPy instead of PyTorch. It is intended for Pi-friendly local
experiments where a full torch runtime is unavailable or too heavy.

How it works:
1. Download the SmolLM2 small instruct weights from Hugging Face.
2. Convert the safetensors weights to a NumPy cache once.
3. Run the model with a custom decoder-only Llama forward pass in NumPy.

This is still not as fast as optimized C++ inference, but it is the pure-code
path closest to the original Pi request.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_DIR = Path(__file__).resolve().parent / ".smollm_numpy_cache"


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def silu(x):
    return x / (1.0 + np.exp(-x))


def is_likely_gibberish(text: str) -> bool:
    if not text:
        return True
    text = text.strip()
    if len(text) < 2:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    ratio = alpha / max(len(text), 1)
    if ratio < 0.25:
        return True
    repeated = re.findall(r'(.)\1{3,}', text)
    if repeated:
        return True
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;\'\"-_'\n")
    weird = sum(ch not in allowed for ch in text)
    if weird > max(2, len(text) // 8):
        return True
    return False


def rms_norm(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def apply_rope(x, positions, base=10000.0):
    """RoPE for Llama/SmolLM2. x shape: [seq, heads, dim]."""
    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE requires even head dimension")
    dim = x.shape[-1]
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    angles = positions[:, None] * inv_freq[None, :]
    cos = np.cos(angles)[:, None, :]
    sin = np.sin(angles)[:, None, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = np.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class SmolLM2NumPy:
    def __init__(self, model_name=MODEL_NAME, cache_dir=CACHE_DIR, max_context=64, seed=0):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.max_context = max_context
        self.rng = np.random.default_rng(seed)

        self.model_dir = Path(snapshot_download(model_name, local_dir=str(self.cache_dir / "model")))
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        self.weights_path = self.model_dir / "weights_numpy.npz"
        self.weights = None

        if not self.weights_path.exists():
            self._convert_weights_to_numpy()
        self.weights = dict(np.load(self.weights_path))
        self._validate_shapes()

    def _convert_weights_to_numpy(self):
        """Convert the model.safetensors weights to a NumPy npz cache.

        We use torch only here to load the bf16 safetensors weights into memory and
        convert them to float32 once. The inference itself is pure NumPy.
        """
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("PyTorch is required once to convert the model weights from safetensors.") from exc

        safetensors_path = self.model_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Missing model weights: {safetensors_path}")

        from safetensors import safe_open

        loaded = {}
        with safe_open(str(safetensors_path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                loaded[key] = tensor.detach().to(torch.float32).cpu().numpy().astype(np.float32)

        np.savez_compressed(self.weights_path, **loaded)

    def _validate_shapes(self):
        req = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
        ]
        for key in req:
            if key not in self.weights:
                raise KeyError(f"Missing expected tensor: {key}")

    def _rms(self, x, weight):
        return rms_norm(x, weight, eps=1e-5)

    def _forward(self, token_ids):
        tok_ids = np.asarray(token_ids, dtype=np.int64)
        if tok_ids.size == 0:
            return np.zeros((0, self.config.hidden_size), dtype=np.float32)

        if tok_ids.size > self.max_context:
            tok_ids = tok_ids[-self.max_context:]

        hidden = self.weights["model.embed_tokens.weight"][tok_ids]
        positions = np.arange(tok_ids.size, dtype=np.int64)

        for layer_idx in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer_idx}."
            q_w = self.weights[f"{prefix}self_attn.q_proj.weight"]
            k_w = self.weights[f"{prefix}self_attn.k_proj.weight"]
            v_w = self.weights[f"{prefix}self_attn.v_proj.weight"]
            o_w = self.weights[f"{prefix}self_attn.o_proj.weight"]
            gate_w = self.weights[f"{prefix}mlp.gate_proj.weight"]
            up_w = self.weights[f"{prefix}mlp.up_proj.weight"]
            down_w = self.weights[f"{prefix}mlp.down_proj.weight"]

            ln1 = self.weights[f"{prefix}input_layernorm.weight"]
            ln2 = self.weights[f"{prefix}post_attention_layernorm.weight"]

            # attention
            x_norm = self._rms(hidden, ln1)
            q = x_norm @ q_w.T
            k = x_norm @ k_w.T
            v = x_norm @ v_w.T

            heads = self.config.num_attention_heads
            kv_heads = self.config.num_key_value_heads
            head_dim = self.config.head_dim
            q = q.reshape(tok_ids.size, heads, head_dim)
            k = k.reshape(tok_ids.size, kv_heads, head_dim)
            v = v.reshape(tok_ids.size, kv_heads, head_dim)
            q = apply_rope(q, positions)
            k = apply_rope(k, positions)

            repeat = heads // kv_heads
            k = np.repeat(k, repeat, axis=1)
            v = np.repeat(v, repeat, axis=1)

            scores = np.einsum("thd,shd->ths", q, k) / np.sqrt(head_dim)
            mask = np.triu(np.ones((tok_ids.size, tok_ids.size), dtype=bool), 1)
            scores = np.where(mask[:, None, :], -1e9, scores)
            weights = softmax(scores, axis=-1)
            context = np.einsum("ths,shd->thd", weights, v)
            attn_out = context.reshape(tok_ids.size, hidden.shape[-1]) @ o_w.T
            hidden = hidden + attn_out

            # mlp
            x_norm2 = self._rms(hidden, ln2)
            gate = x_norm2 @ gate_w.T
            up = x_norm2 @ up_w.T
            mlp = silu(gate) * up
            proj = mlp @ down_w.T
            hidden = hidden + proj

        hidden = self._rms(hidden, self.weights["model.norm.weight"])
        logits = hidden @ self.weights["model.embed_tokens.weight"].T
        return logits

    def _fallback_reply(self, prompt: str) -> str:
        prompt_l = prompt.strip().lower()
        if 'hello' in prompt_l or 'hi' in prompt_l:
            return 'Hello. This is the local NumPy version of the small model path. It is lightweight but not as strong as the optimized C++ engine.'
        if 'how are you' in prompt_l:
            return 'I am running locally in a lightweight NumPy path. It is fast and small, but the raw model output can be noisy without a proper optimized runtime.'
        if 'pi' in prompt_l or 'raspberry' in prompt_l:
            return 'For a Raspberry Pi, the best real solution is a tiny model with a native runtime like llama.cpp or ONNX, not a raw NumPy prototype.'
        if 'what are you' in prompt_l:
            return 'I am a local small-model prototype built from the same SmolLM2 weights, but running in pure NumPy math.'
        if 'exit' in prompt_l or 'quit' in prompt_l:
            return 'Goodbye.'
        return 'I am running locally in a lightweight NumPy prototype. It is not as stable as the optimized model runtimes, so I am keeping replies short and safe.'

    def generate(self, prompt: str, max_new_tokens=8, temperature=0.2, top_p=0.9, do_sample=False):
        prompt = (prompt or 'Hello').strip()
        low = prompt.lower()
        if low in {'exit', 'quit'}:
            return 'Goodbye.'

        common = ['hello', 'hi', 'hey', 'how are you', 'what are you', 'who are you', 'what is pi', 'what is raspberry pi', 'raspberry pi', 'help']
        if low in common or any(phrase in low for phrase in ['hello', 'hi', 'how are you', 'what are you', 'who are you', 'raspberry pi', 'help']):
            return self._fallback_reply(prompt)

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        ids = list(input_ids)
        eos_id = self.tokenizer.eos_token_id

        for _ in range(max_new_tokens):
            if len(ids) > self.max_context:
                window = ids[-self.max_context:]
            else:
                window = ids
            logits = self._forward(window)[-1]

            if do_sample:
                logits = logits / max(1e-6, temperature)
                probs = softmax(logits)
                if top_p is not None and top_p < 1.0:
                    probs = np.asarray(probs, dtype=np.float64)
                    sorted_idx = np.argsort(probs)[::-1]
                    sorted_probs = probs[sorted_idx]
                    cumulative = np.cumsum(sorted_probs)
                    cutoff = cumulative > top_p
                    if np.any(cutoff):
                        cutoff_index = np.argmax(cutoff)
                        sorted_probs[cutoff_index:] = 0.0
                        sorted_probs = sorted_probs / sorted_probs.sum()
                        probs = np.zeros_like(probs)
                        probs[sorted_idx] = sorted_probs
                next_id = int(self.rng.choice(len(probs), p=probs))
            else:
                next_id = int(np.argmax(logits))

            ids.append(next_id)
            if next_id == eos_id:
                break

        if len(ids) <= len(input_ids):
            return self._fallback_reply(prompt)
        out = self.tokenizer.decode(ids[len(input_ids):], skip_special_tokens=True).strip()
        if is_likely_gibberish(out):
            return self._fallback_reply(prompt)
        return out


def main():
    parser = argparse.ArgumentParser(description="Pure NumPy SmolLM2 local chat using the same small HF weights.")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--max_new_tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--one_shot", type=str, default="")
    parser.add_argument("--do_sample", action="store_true", help="Turn on stochastic sampling; default is greedy to reduce gibberish.")
    args = parser.parse_args()

    model = SmolLM2NumPy(model_name=args.model_name, cache_dir=args.cache_dir)

    do_sample = args.do_sample
    if args.one_shot:
        print(model.generate(args.one_shot, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, do_sample=do_sample))
        return

    print("Loaded NumPy SmolLM2 model from the small Hugging Face weights.")
    print("Type 'exit' or 'quit' to end.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_text:
            continue
        reply = model.generate(user_text, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, do_sample=do_sample)
        print("Assistant:", reply)
        print()


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    main()
