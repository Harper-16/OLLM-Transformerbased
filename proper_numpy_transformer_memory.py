#!/usr/bin/env python3
"""Proper transformer-style architecture with local long-term memory.

This is a realistic low-footprint design inspired by GPT-style transformers, but
built for local, small-device experimentation with NumPy and SQLite memory.

It includes:
- tokenizer
- embeddings + positional encodings
- causal self-attention blocks
- feed-forward MLP
- long-term memory store (SQLite)
- retrieval from saved memory chunks
- next-word prediction loop

This is still not a full production LLM, but it is much closer to a proper
transformer architecture than the earlier toy prototypes.
"""

import argparse
import json
import math
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from urllib import request, error

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9']+")


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


class SQLiteMemory:
    def __init__(self, db_path: str = '.memory.sqlite'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                tags TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def embed_text(self, text: str):
        words = WORD_RE.findall(text.lower())
        if not words:
            return np.zeros(64, dtype=np.float32)
        count = Counter(words)
        vec = np.zeros(64, dtype=np.float32)
        for i, w in enumerate(sorted(count.keys())[:64]):
            vec[i] = float(count[w])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add(self, text: str, source: str = 'local', tags: str = 'general'):
        clean = re.sub(r'\s+', ' ', text).strip()
        if len(clean) < 25:
            return
        emb = self.embed_text(clean)
        self.conn.execute(
            "INSERT INTO memory (text, source, tags, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (clean, source, tags, time.strftime('%Y-%m-%dT%H:%M:%S'), json.dumps(emb.tolist())),
        )
        self.conn.commit()

    def retrieve(self, query: str, limit: int = 5):
        q = self.embed_text(query)
        rows = self.conn.execute('SELECT id, text, source, embedding FROM memory').fetchall()
        scored = []
        for row in rows:
            _, text, source, payload = row
            emb = np.asarray(json.loads(payload), dtype=np.float32)
            target = np.zeros_like(q, dtype=np.float32)
            n = min(q.size, emb.size)
            target[:n] = emb[:n]
            sim = float(np.dot(q, target)) if emb.size else 0.0
            scored.append((sim, text, source))
        scored.sort(reverse=True)
        return scored[:limit]

    def ingest_files(self, root: str):
        path = Path(root)
        if not path.exists():
            return 0
        count = 0
        for f in sorted(path.rglob('*')):
            if not f.is_file():
                continue
            if f.suffix.lower() not in {'.txt', '.md', '.json', '.csv', '.log', '.py'}:
                continue
            try:
                txt = f.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            chunks = split_chunks(txt, max_words=180)
            for ch in chunks:
                self.add(ch, source=str(f))
                count += 1
        return count


def split_chunks(text: str, max_words: int = 180):
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(' '.join(words[i:i + max_words]))
    return chunks


class WordTokenizer:
    def __init__(self, texts=None):
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        if texts:
            self.fit(texts)

    def fit(self, texts):
        counts = Counter()
        for t in texts:
            counts.update(WORD_RE.findall(t.lower()))
        for word, _ in counts.most_common(800):
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word[len(self.id_to_word)] = word

    def encode(self, text: str):
        words = WORD_RE.findall(text.lower())
        if not words:
            return np.array([1], dtype=np.int64)
        return np.array([self.word_to_id.get(w, 1) for w in words], dtype=np.int64)

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(int(i), '<UNK>') for i in ids)


class LayerNorm:
    def __init__(self, dim: int):
        self.g = np.ones(dim, dtype=np.float32)
        self.b = np.zeros(dim, dtype=np.float32)

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * self.g + self.b


class MiniTransformerBlock:
    def __init__(self, dim: int, heads: int, ff_dim: int, seed: int):
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        rng = np.random.default_rng(seed)
        self.ln1 = LayerNorm(dim)
        self.qw = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)
        self.kw = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)
        self.vw = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)
        self.ow = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)

        self.ln2 = LayerNorm(dim)
        self.ff1 = rng.normal(0.0, 0.1, size=(dim, ff_dim)).astype(np.float32)
        self.ff2 = rng.normal(0.0, 0.1, size=(ff_dim, dim)).astype(np.float32)

    def forward(self, x):
        x_norm = self.ln1(x)
        q = x_norm @ self.qw
        k = x_norm @ self.kw
        v = x_norm @ self.vw

        seq = x.shape[0]
        q = q.reshape(seq, self.heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(seq, self.heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(seq, self.heads, self.head_dim).transpose(1, 0, 2)

        attn = (q @ k.transpose(0, 2, 1)) / math.sqrt(self.head_dim)
        mask = np.triu(np.ones((seq, seq), dtype=np.float32), 1)
        attn = np.where(mask == 0, attn, -1e9)
        attn = softmax(attn, axis=-1)
        ctx = attn @ v
        ctx = ctx.transpose(1, 0, 2).reshape(seq, self.dim)
        x = x + ctx @ self.ow

        y = self.ln2(x)
        ff = gelu(y @ self.ff1) @ self.ff2
        x = x + ff
        return x


class MiniTransformerLM:
    def __init__(self, vocab_size: int, dim: int = 64, layers: int = 3, heads: int = 4, context: int = 32, seed: int = 7):
        self.vocab_size = vocab_size
        self.dim = dim
        self.context = context
        rng = np.random.default_rng(seed)
        self.embed = rng.normal(0.0, 0.12, size=(vocab_size, dim)).astype(np.float32)
        self.pos = rng.normal(0.0, 0.05, size=(context, dim)).astype(np.float32)
        self.blocks = [MiniTransformerBlock(dim, heads, dim * 2, seed + i) for i in range(layers)]
        self.norm = LayerNorm(dim)
        self.ff_out = rng.normal(0.0, 0.12, size=(dim, vocab_size)).astype(np.float32)

    def forward(self, token_ids):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.size == 0:
            return np.zeros((0, self.vocab_size), dtype=np.float32)
        if ids.size > self.context:
            ids = ids[-self.context:]
        x = self.embed[ids] + self.pos[: ids.size]
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm(x)
        logits = x @ self.ff_out
        return logits

    def predict_next(self, token_ids, temperature: float = 0.8):
        logits = self.forward(token_ids)[-1]
        logits = logits / max(temperature, 1e-3)
        probs = softmax(logits)
        next_id = int(np.random.choice(self.vocab_size, p=probs))
        return next_id

    def generate(self, prompt_tokens, max_new: int = 12, temperature: float = 0.8):
        ids = list(prompt_tokens)
        for _ in range(max_new):
            next_id = self.predict_next(np.asarray(ids, dtype=np.int64), temperature=temperature)
            if next_id == 0:
                break
            ids.append(next_id)
        return ids


def fetch_url_text(url: str, timeout: int = 20):
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except (error.URLError, error.HTTPError, ValueError):
        return ''
    try:
        text = body.decode('utf-8')
    except Exception:
        text = body.decode('latin-1', errors='ignore')
    return text


def save_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text[:200000], encoding='utf-8', errors='ignore')


def collect_web_text(urls, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for idx, url in enumerate(urls):
        text = fetch_url_text(url)
        if not text:
            continue
        text = re.sub(r'\s+', ' ', text)
        if len(text) < 200:
            continue
        save_text(out / f'web_{idx}.txt', text)


def build_corpus_from_dir(data_dir: str):
    root = Path(data_dir)
    chunks = []
    for f in sorted(root.rglob('*')):
        if f.is_file() and f.suffix.lower() in {'.txt', '.md', '.json', '.csv', '.log', '.py'}:
            try:
                txt = f.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            txt = re.sub(r'\s+', ' ', txt)
            if len(txt) > 100:
                chunks.append(txt)
    return '\n\n'.join(chunks)


def train_and_answer(prompt: str, corpus: str, memory_db: str = '.memory.sqlite', max_new: int = 12):
    memory = SQLiteMemory(memory_db)
    for chunk in split_chunks(corpus, max_words=180):
        memory.add(chunk, source='corpus')

    texts = [corpus]
    tok = WordTokenizer(texts)
    if len(tok.id_to_word) < 2:
        return 'No usable vocabulary.'

    model = MiniTransformerLM(vocab_size=len(tok.id_to_word), dim=64, layers=3, heads=4, context=32, seed=7)
    ids = tok.encode(corpus)
    if len(ids) < 5:
        return 'Not enough data.'

    retrieved = memory.retrieve(prompt, limit=3)
    memory_context = ' '.join(text for _, text, _ in retrieved)
    combined = (prompt + ' ' + memory_context + ' ' + corpus)[:4000]
    base_ids = tok.encode(combined)
    generated = model.generate(base_ids[:32], max_new=max_new, temperature=0.8)
    answer = tok.decode(generated)
    return answer


def main():
    parser = argparse.ArgumentParser(description='Proper transformer-style local model with SQLite memory and web ingestion.')
    parser.add_argument('--prompt', type=str, default='hello there')
    parser.add_argument('--max_new', type=int, default=12)
    parser.add_argument('--url-file', type=str, default='')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--db', type=str, default='.memory.sqlite')
    args = parser.parse_args()

    urls = [
        'https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)',
        'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
        'https://en.wikipedia.org/wiki/Raspberry_Pi',
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
    ]

    if args.url_file:
        with open(args.url_file, 'r', encoding='utf-8', errors='ignore') as f:
            urls = [line.strip() for line in f if line.strip()]

    collect_web_text(urls, args.data_dir)
    corpus = build_corpus_from_dir(args.data_dir)
    answer = train_and_answer(args.prompt, corpus, memory_db=args.db, max_new=args.max_new)
    print('Prompt:', args.prompt)
    print('Answer:', answer[:400])


if __name__ == '__main__':
    main()
