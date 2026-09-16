#!/usr/bin/env python3
"""Tiny transformer for Pi Zero W.

This is a realistic low-memory, low-CPU model for a Raspberry Pi Zero W.
It uses:
- pure NumPy
- causal self-attention
- a tiny transformer block
- character-level training on local text

It is not a production assistant. It is a compact learning prototype designed
for very small hardware and a small dataset.
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def rms_norm(x, weight, eps=1e-5):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


class PiZeroTokenizer:
    def __init__(self, text=None):
        self.vocab = {}
        if text is not None:
            self.fit(text)

    def fit(self, text):
        chars = sorted(set(text))
        if len(chars) > 256:
            chars = chars[:256]
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.inv = {i: ch for ch, i in self.vocab.items()}
        self.unk = len(self.vocab)
        self.vocab['<UNK>'] = self.unk
        self.inv[self.unk] = '<UNK>'

    def encode(self, text):
        if not self.vocab:
            raise ValueError('Tokenizer has no vocabulary. Call fit() first.')
        out = []
        for ch in text:
            out.append(self.vocab.get(ch, self.unk))
        return np.asarray(out, dtype=np.int64)

    def decode(self, tokens):
        if not self.inv:
            return ''
        return ''.join(self.inv.get(int(t), '<UNK>') for t in tokens)


class TinyAttentionBlock:
    def __init__(self, d_model, n_heads, ff_dim, rng):
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model

        self.ln1_w = np.ones(d_model, dtype=np.float32)
        self.ln1_b = np.zeros(d_model, dtype=np.float32)
        self.qkv_w = rng.normal(0.0, 0.15, size=(d_model, d_model * 3)).astype(np.float32)
        self.qkv_b = np.zeros(d_model * 3, dtype=np.float32)
        self.o_w = rng.normal(0.0, 0.15, size=(d_model, d_model)).astype(np.float32)
        self.o_b = np.zeros(d_model, dtype=np.float32)

        self.ln2_w = np.ones(d_model, dtype=np.float32)
        self.ln2_b = np.zeros(d_model, dtype=np.float32)
        self.ff1_w = rng.normal(0.0, 0.15, size=(d_model, ff_dim)).astype(np.float32)
        self.ff1_b = np.zeros(ff_dim, dtype=np.float32)
        self.ff2_w = rng.normal(0.0, 0.15, size=(ff_dim, d_model)).astype(np.float32)
        self.ff2_b = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        x_ln = x.copy()
        for i in range(x.shape[0]):
            mean = x[i].mean()
            var = x[i].var() + 1e-5
            x_ln[i] = (x[i] - mean) / np.sqrt(var) * self.ln1_w + self.ln1_b

        qkv = x_ln @ self.qkv_w + self.qkv_b
        q, k, v = np.split(qkv, 3, axis=-1)

        seq = x.shape[0]
        q = q.reshape(seq, self.n_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq, self.n_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, self.n_heads, self.head_dim).transpose(1, 0, 2)

        weight = np.matmul(q, np.swapaxes(k, -1, -2)) / math.sqrt(self.head_dim)
        mask = np.triu(np.ones((seq, seq), dtype=np.float32), 1)
        weight = np.where(mask == 0, weight, -1e9)
        weight = softmax(weight, axis=-1)
        ctx = np.matmul(weight, v)
        ctx = ctx.transpose(1, 0, 2).reshape(seq, self.d_model)
        out = ctx @ self.o_w + self.o_b
        x = x + out

        y = x.copy()
        for i in range(x.shape[0]):
            mean = y[i].mean()
            var = y[i].var() + 1e-5
            y[i] = (y[i] - mean) / np.sqrt(var) * self.ln2_w + self.ln2_b

        ff = gelu(y @ self.ff1_w + self.ff1_b) @ self.ff2_w + self.ff2_b
        x = x + ff
        return x


class TinyPiTransformer:
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_layers=2, max_ctx=64, seed=7):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_ctx = max_ctx
        self.rng = np.random.default_rng(seed)

        self.Wte = self.rng.normal(0.0, 0.15, size=(vocab_size, d_model)).astype(np.float32)
        self.Wpe = self.rng.normal(0.0, 0.1, size=(max_ctx, d_model)).astype(np.float32)
        self.blocks = [TinyAttentionBlock(d_model, n_heads, d_model * 2, self.rng) for _ in range(n_layers)]
        self.ln_f_w = np.ones(d_model, dtype=np.float32)
        self.ln_f_b = np.zeros(d_model, dtype=np.float32)

    def forward(self, token_ids):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        seq = token_ids.shape[0]
        if seq > self.max_ctx:
            token_ids = token_ids[-self.max_ctx:]
            seq = token_ids.shape[0]

        x = self.Wte[token_ids] + self.Wpe[:seq]
        for block in self.blocks:
            x = block.forward(x)

        x = x.copy()
        for i in range(x.shape[0]):
            mean = x[i].mean()
            var = x[i].var() + 1e-5
            x[i] = (x[i] - mean) / np.sqrt(var) * self.ln_f_w + self.ln_f_b

        logits = x @ self.Wte.T
        return logits

    def generate(self, prompt, tokenizer, max_new=20, temperature=0.8, seed=None):
        if seed is not None:
            np.random.seed(seed)

        ids = tokenizer.encode(prompt).astype(np.int64).tolist()
        for _ in range(max_new):
            window = ids[-self.max_ctx:]
            logits = self.forward(np.asarray(window, dtype=np.int64))[-1]
            logits = logits / max(temperature, 1e-6)
            probs = softmax(logits)
            next_id = int(np.random.choice(len(probs), p=probs))
            ids.append(next_id)
            if next_id == 0:
                break
        return tokenizer.decode(ids[len(tokenizer.encode(prompt)):])

    def train_step(self, token_ids, targets, lr=0.01):
        logits = self.forward(token_ids)
        logits = logits.astype(np.float64)
        probs = softmax(logits, axis=-1)
        loss = -np.log(probs[np.arange(len(token_ids)), targets] + 1e-8).mean()

        grad_logits = probs.copy()
        grad_logits[np.arange(len(token_ids)), targets] -= 1.0
        grad_logits /= len(token_ids)

        # Train only the embedding layer and position embeddings. This keeps the
        # implementation lightweight and stable on a Pi Zero W.
        grad_Wte = np.zeros_like(self.Wte)
        grad_Wpe = np.zeros_like(self.Wpe)

        for i, t in enumerate(token_ids):
            grad_emb = grad_logits[i] @ self.Wte
            grad_Wte[t] += grad_emb
            if i < self.max_ctx:
                grad_Wpe[i] += grad_emb

        self.Wte -= lr * grad_Wte
        self.Wpe -= lr * grad_Wpe
        return float(loss)


def load_corpus(paths):
    texts = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.is_file() and f.suffix.lower() in {'.txt', '.md', '.py', '.json', '.csv'}:
                    try:
                        texts.append(f.read_text(encoding='utf-8', errors='ignore'))
                    except Exception:
                        pass
        elif p.is_file():
            try:
                texts.append(p.read_text(encoding='utf-8', errors='ignore'))
            except Exception:
                pass
    if not texts:
        texts = [
            'hello world this is a tiny transformer for raspberry pi zero w. ',
            'we use numpy attention and a small dataset so it stays lightweight. ',
            'the model learns simple patterns and generates short text. '
        ]
    return '\n'.join(texts)


def train_model(corpus, steps=50, d_model=32, n_layers=2, max_ctx=32, seed=7):
    tokenizer = PiZeroTokenizer(corpus)
    model = TinyPiTransformer(vocab_size=len(tokenizer.vocab), d_model=d_model, n_layers=n_layers, max_ctx=max_ctx, seed=seed)
    ids = tokenizer.encode(corpus)

    # train in a cheap teacher-forced way
    losses = []
    for step in range(steps):
        start = random.randrange(0, max(1, len(ids) - max_ctx))
        snippet = ids[start:start + max_ctx]
        if len(snippet) < 2:
            continue
        x = snippet[:-1]
        y = snippet[1:]
        loss = model.train_step(x, y, lr=0.01)
        losses.append(loss)

    return model, tokenizer, losses


def main():
    parser = argparse.ArgumentParser(description='Tiny NumPy transformer for a Raspberry Pi Zero W.')
    parser.add_argument('--data', type=str, default='data', help='Folder or file containing training text.')
    parser.add_argument('--steps', type=int, default=80, help='Training steps. Keep it low on Pi Zero W.')
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--max_ctx', type=int, default=32)
    parser.add_argument('--prompt', type=str, default='hello')
    parser.add_argument('--max_new', type=int, default=15)
    args = parser.parse_args()

    corpus = load_corpus([args.data])
    model, tokenizer, losses = train_model(corpus, steps=args.steps, d_model=args.d_model, n_layers=args.n_layers, max_ctx=args.max_ctx, seed=7)

    text = model.generate(args.prompt, tokenizer, max_new=args.max_new, temperature=0.8)
    print(f'losses: {losses[:5]} ... final={losses[-1] if losses else 0.0}')
    print(f'Prompt: {args.prompt}')
    print(f'Generated: {text}')


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
