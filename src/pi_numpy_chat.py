#!/usr/bin/env python3
"""Pure NumPy local chat engine tuned for Raspberry Pi.

This is intentionally different from pi_chat.py:
- no torch
- no transformers
- no llama.cpp
- all math is done with NumPy + hand-written attention/MLP logic
- optimized for CPU and small memory use

It gives you a fast local chat loop with:
1. rule-based replies for common prompts
2. a tiny decoder-only transformer forward pass in NumPy
3. low-CPU, no external model download required

Usage:
    python3 pi_numpy_chat.py
    python3 pi_numpy_chat.py --max_new_tokens 30 --temperature 0.9
"""

import argparse
import math
import os
import re
from datetime import datetime

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CharTokenizer:
    """Very small byte-style tokenizer for a local Pi-friendly model."""

    def __init__(self):
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_'\"/:;()[]{}\n"
        self.vocab = {ch: i for i, ch in enumerate(alphabet)}
        self.rev = {i: ch for ch, i in self.vocab.items()}
        self.unk_id = len(self.vocab)

    def encode(self, text: str):
        tokens = []
        for ch in text:
            tokens.append(self.vocab.get(ch, self.unk_id))
        return np.asarray(tokens, dtype=np.int64)

    def decode(self, ids):
        out = []
        for idx in ids:
            ch = self.rev.get(int(idx), " ")
            out.append(ch)
        return "".join(out)


class TinyTransformerBlock:
    def __init__(self, d_model, n_heads, ff_dim, rng):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ff_dim = ff_dim

        self.ln1_w = np.ones(d_model, dtype=np.float32)
        self.ln1_b = np.zeros(d_model, dtype=np.float32)
        self.qkv_w = rng.normal(0.0, 0.12, size=(d_model, d_model * 3)).astype(np.float32)
        self.qkv_b = np.zeros(d_model * 3, dtype=np.float32)
        self.out_w = rng.normal(0.0, 0.12, size=(d_model, d_model)).astype(np.float32)
        self.out_b = np.zeros(d_model, dtype=np.float32)

        self.ln2_w = np.ones(d_model, dtype=np.float32)
        self.ln2_b = np.zeros(d_model, dtype=np.float32)
        self.fc1_w = rng.normal(0.0, 0.12, size=(d_model, ff_dim)).astype(np.float32)
        self.fc1_b = np.zeros(ff_dim, dtype=np.float32)
        self.fc2_w = rng.normal(0.0, 0.12, size=(ff_dim, d_model)).astype(np.float32)
        self.fc2_b = np.zeros(d_model, dtype=np.float32)

    @staticmethod
    def layer_norm(x, w, b):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x):
        # Multi-head self-attention
        x_norm = self.layer_norm(x, self.ln1_w, self.ln1_b)
        qkv = x_norm @ self.qkv_w + self.qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)

        seq_len = x.shape[0]
        q = q.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq_len, self.n_heads, self.head_dim).transpose(1, 0, 2)

        attn_scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
        mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1)
        mask = mask == 1
        attn_scores[:, mask] = -1e9
        attn_weights = softmax(attn_scores, axis=-1)
        context = attn_weights @ v
        context = context.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        x = x + context @ self.out_w + self.out_b

        # FFN
        x_norm = self.layer_norm(x, self.ln2_w, self.ln2_b)
        hidden = self.gelu(x_norm @ self.fc1_w + self.fc1_b)
        x = x + hidden @ self.fc2_w + self.fc2_b
        return x


class TinyLocalModel:
    def __init__(self, vocab_size=128, d_model=64, n_heads=4, n_layers=2, ctx=32, seed=7):
        self.tokenizer = CharTokenizer()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ctx = ctx
        self.rng = np.random.default_rng(seed)

        self.embed = self.rng.normal(0.0, 0.12, size=(self.vocab_size, d_model)).astype(np.float32)
        self.pos = self.rng.normal(0.0, 0.12, size=(ctx, d_model)).astype(np.float32)
        self.ln_f_w = np.ones(d_model, dtype=np.float32)
        self.ln_f_b = np.zeros(d_model, dtype=np.float32)

        self.blocks = [
            TinyTransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=d_model * 2, rng=self.rng)
            for _ in range(n_layers)
        ]

    @staticmethod
    def layer_norm(x, w, b):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5) * w + b

    def forward(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.size == 0:
            return np.zeros((0, self.d_model), dtype=np.float32)

        if token_ids.size > self.ctx:
            token_ids = token_ids[-self.ctx:]

        x = self.embed[token_ids] + self.pos[: token_ids.size]
        for block in self.blocks:
            x = block.forward(x)
        x = self.layer_norm(x, self.ln_f_w, self.ln_f_b)
        logits = x @ self.embed.T
        return logits

    def generate(self, prompt: str, max_new_tokens=20, temperature=0.9):
        ids = self.tokenizer.encode(prompt)
        generated = list(ids)
        for _ in range(max_new_tokens):
            if len(generated) > self.ctx:
                window = generated[-self.ctx:]
            else:
                window = generated

            logits = self.forward(window)[-1]
            logits = logits.astype(np.float32)
            if temperature <= 0:
                next_id = int(np.argmax(logits))
            else:
                logits = logits / temperature
                logits = logits - np.max(logits)
                probs = np.exp(logits)
                probs = probs / probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))

            generated.append(next_id)
            ch = self.tokenizer.decode([next_id])
            if not ch or ch.isspace() and len(generated) > 4:
                # Mild stop when generation drifts into whitespace noise.
                break

        text = self.tokenizer.decode(generated[len(ids):])
        return text.strip()


class RuleAssistant:
    def __init__(self):
        self.responses = [
            ("hello", "Hello! I am a pure NumPy local assistant running on CPU. I do not use PyTorch or Transformers."),
            ("hi", "Hi! I am a local NumPy assistant. I keep things lightweight and efficient on a Pi."),
            ("how are you", "I am running efficiently in plain NumPy. Fast, small, and local."),
            ("pi", "This setup is tuned for Raspberry Pi CPU usage: small model, no heavy runtime, no llama.cpp."),
            ("help", "I can help with local AI on a Pi, pure NumPy inference, efficient math, and lightweight model design."),
            ("what are you", "I am a tiny local language engine built with pure NumPy math and logic."),
            ("time", "The current time is {}."),
            ("date", "Today is {}."),
            ("goodbye", "Goodbye! Keep it local and efficient."),
            ("exit", "Goodbye! Keep it local and efficient."),
            ("quit", "Goodbye! Keep it local and efficient."),
        ]

    def respond(self, text: str):
        lower = text.strip().lower()
        if not lower:
            return ""

        for key, reply in self.responses:
            if key in lower:
                if "{}" in reply:
                    return reply.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                return reply

        if "math" in lower or "numpy" in lower:
            return "NumPy is great here because matrix ops are cheap, vectorized, and CPU-friendly. We avoid heavyweight runtimes and keep inference local."

        if "efficient" in lower or "fast" in lower:
            return "The fastest Pi path is: small context, no giant weights, no warm Python overhead, and plain NumPy loops for attention/MLP."

        if "model" in lower or "ai" in lower:
            return "The best local Pi approach is a tiny transformer or a compact small language model, with CPU-friendly inference and low memory use."

        return None


class NumpyPiChat:
    def __init__(self, seed=7):
        self.model = TinyLocalModel(seed=seed)
        self.rules = RuleAssistant()

    def generate_reply(self, user_text: str, max_new_tokens: int = 30, temperature: float = 0.9):
        rule_reply = self.rules.respond(user_text)
        if rule_reply is not None:
            return rule_reply

        # Fallback: tiny transformer text generation using a pure NumPy forward pass.
        prompt = user_text.strip() or "hello"
        generated = self.model.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        if not generated:
            generated = "I am running locally with NumPy and pure math. Ask me for Pi tuning, model design, or efficient inference."
        return generated


def main():
    parser = argparse.ArgumentParser(description='Pure NumPy local chat optimized for Raspberry Pi.')
    parser.add_argument('--max_new_tokens', type=int, default=28, help='Tokens to generate per answer.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature for generation.')
    parser.add_argument('--seed', type=int, default=7, help='Random seed for deterministic behavior.')
    parser.add_argument('--one_shot', type=str, default='', help='Optional single prompt instead of interactive chat.')
    args = parser.parse_args()

    chat = NumpyPiChat(seed=args.seed)

    if args.one_shot:
        print(chat.generate_reply(args.one_shot, max_new_tokens=args.max_new_tokens, temperature=args.temperature))
        return

    print('Pure NumPy Pi chat started. Type "exit" or "quit" to end.')
    print('This version avoids PyTorch, Transformers, and llama.cpp.')

    while True:
        user_text = input('You: ').strip()
        if user_text.lower() in {'exit', 'quit'}:
            print('Goodbye!')
            break
        if not user_text:
            continue

        reply = chat.generate_reply(user_text, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print('Assistant:', reply)
        print()


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    np.set_printoptions(suppress=True)
    main()
