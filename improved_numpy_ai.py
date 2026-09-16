#!/usr/bin/env python3
"""Improved lightweight NumPy AI (practical upgrade)

Goals implemented in this script:
- Robust HTML/text cleaning and extraction when downloading web pages
- Word-level tokenizer with top-N vocabulary and UNK/PAD tokens
- Simple context-averaging encoder (fast on tiny devices) + output projection
- Adam optimizer training embeddings + output projection (cheap, effective)
- Long-term memory retrieval: include retrieved memory embeddings in context
- Training loop with minibatches sampled from corpus
- Sampling/generation with temperature, top-k and top-p options

This is still a small local model (not ChatGPT), but it is a pragmatic and
trainable NumPy-based neural model designed to be more useful than earlier
prototypes while remaining lightweight enough to run on small hardware.
"""

import argparse
import math
import re
import time
from collections import Counter, deque
from html.parser import HTMLParser
from pathlib import Path
from typing import List
from urllib import request, error

import numpy as np


# ---------------------- Text extraction / cleaning ----------------------

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "canvas"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "canvas"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            s = data.strip()
            if s:
                self.chunks.append(s)

    def get_text(self):
        return " ".join(self.chunks)


def fetch_text_from_url(url: str, timeout: int = 15) -> str:
    try:
        req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except Exception:
        return ""
    try:
        text = raw.decode("utf-8")
    except Exception:
        try:
            text = raw.decode("latin-1")
        except Exception:
            return ""
    parser = TextExtractor()
    parser.feed(text)
    out = parser.get_text()
    # basic cleanup
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"https?://\S+", "", out)
    return out.strip()


def load_local_texts(folder: Path, min_chars: int = 50) -> List[str]:
    texts = []
    if not folder.exists():
        return texts
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".txt", ".md", ".py", ".json", ".csv"}:
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        t = re.sub(r"\s+", " ", t).strip()
        if len(t) >= min_chars:
            texts.append(t)
    return texts


# ---------------------- Tokenizer ----------------------

WORD_RE = re.compile(r"[A-Za-z0-9']+")


class WordTokenizer:
    def __init__(self, max_vocab: int = 10000):
        self.max_vocab = max_vocab
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build(self, texts: List[str]):
        counter = Counter()
        for t in texts:
            counter.update(WORD_RE.findall(t.lower()))
        for w, _ in counter.most_common(self.max_vocab - 2):
            if w in self.word_to_id:
                continue
            idx = len(self.id_to_word)
            self.word_to_id[w] = idx
            self.id_to_word[idx] = w
        self.vocab_size = len(self.id_to_word)

    def encode(self, text: str) -> np.ndarray:
        words = WORD_RE.findall(text.lower())
        if not words:
            return np.array([1], dtype=np.int64)
        return np.array([self.word_to_id.get(w, 1) for w in words], dtype=np.int64)

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id_to_word.get(int(i), "<UNK>") for i in ids)


# ---------------------- Simple context model (fast) ----------------------

class SimpleContextModel:
    """Model computes a context vector as the mean of recent embeddings (and
    optionally memory embeddings) and maps it to logits with an output matrix.

    Trains embeddings and output matrix with Adam.
    """

    def __init__(self, vocab_size: int, dim: int = 128, ctx: int = 8, seed: int = 7):
        self.vocab_size = vocab_size
        self.dim = dim
        self.ctx = ctx
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0.0, 0.2, size=(vocab_size, dim)).astype(np.float32)
        self.out = rng.normal(0.0, 0.2, size=(dim, vocab_size)).astype(np.float32)

        # Adam states
        self.m_emb = np.zeros_like(self.emb)
        self.v_emb = np.zeros_like(self.emb)
        self.m_out = np.zeros_like(self.out)
        self.v_out = np.zeros_like(self.out)
        self.opt_t = 0

    def context_vector(self, token_ids: List[int], memory_vecs: List[np.ndarray] = None) -> np.ndarray:
        if len(token_ids) == 0:
            ctx_vec = np.zeros(self.dim, dtype=np.float32)
        else:
            ids = np.array(token_ids[-self.ctx:], dtype=np.int64)
            ctx_vec = self.emb[ids].mean(axis=0)
        if memory_vecs:
            mv = np.mean(np.stack(memory_vecs, axis=0), axis=0)
            # weighted mix
            ctx_vec = 0.7 * ctx_vec + 0.3 * mv
        return ctx_vec

    def logits_from_context(self, ctx_vec: np.ndarray) -> np.ndarray:
        return ctx_vec @ self.out

    def predict_next(self, token_ids: List[int], memory_vecs: List[np.ndarray] = None, temperature: float = 1.0, top_k: int = 40, top_p: float = 0.9) -> int:
        ctx = self.context_vector(token_ids, memory_vecs)
        logits = self.logits_from_context(ctx)
        # temperature + top-k/top-p filtering
        if temperature <= 0:
            return int(np.argmax(logits))
        logits = logits / max(1e-6, temperature)
        # top-k
        if top_k and top_k < logits.size:
            idx = np.argpartition(-logits, top_k)[:top_k]
            mask = np.ones_like(logits, dtype=bool)
            mask[idx] = False
            logits[mask] = -1e9
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()        # top-p
        if top_p < 1.0:
            sidx = np.argsort(-probs)
            cum = np.cumsum(probs[sidx])
            keep = cum <= top_p
            if not keep.all():
                cutoff = np.argmax(cum > top_p)
                probs[sidx[cutoff + 1:]] = 0.0
                probs = probs / probs.sum()
        return int(np.random.choice(self.vocab_size, p=probs))

    def adam_step(self, grad_emb, grad_out, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.opt_t += 1
        # out
        self.m_out = beta1 * self.m_out + (1 - beta1) * grad_out
        self.v_out = beta2 * self.v_out + (1 - beta2) * (grad_out ** 2)
        m_hat = self.m_out / (1 - beta1 ** self.opt_t)
        v_hat = self.v_out / (1 - beta2 ** self.opt_t)
        self.out -= lr * m_hat / (np.sqrt(v_hat) + eps)
        # emb
        self.m_emb = beta1 * self.m_emb + (1 - beta1) * grad_emb
        self.v_emb = beta2 * self.v_emb + (1 - beta2) * (grad_emb ** 2)
        m_hat_e = self.m_emb / (1 - beta1 ** self.opt_t)
        v_hat_e = self.v_emb / (1 - beta2 ** self.opt_t)
        self.emb -= lr * m_hat_e / (np.sqrt(v_hat_e) + eps)

    def train_batch(self, batch_windows: List[List[int]], tokenizer: WordTokenizer, memory_vectors: List[List[np.ndarray]] = None, lr=1e-3):
        # accumulate grads
        grad_out = np.zeros_like(self.out, dtype=np.float32)
        grad_emb = np.zeros_like(self.emb, dtype=np.float32)
        loss = 0.0
        for i, window in enumerate(batch_windows):
            if len(window) < 2:
                continue
            context = window[:-1]
            target = window[-1]
            mem_vecs = None
            if memory_vectors and i < len(memory_vectors):
                mem_vecs = memory_vectors[i]
            ctx = self.context_vector(context, mem_vecs)
            logits = self.logits_from_context(ctx)
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            loss -= math.log(probs[target] + 1e-12)
            # gradients
            g = probs
            g[target] -= 1.0
            # grad_out: dim x vocab
            grad_out += np.outer(ctx, g).astype(np.float32)
            # gradient to ctx vector: out @ g
            grad_ctx = (self.out @ g).astype(np.float32)
            # distribute equally to embeddings of tokens in context
            if len(context) > 0:
                per_token = grad_ctx / len(context)
                for tid in context:
                    grad_emb[tid] += per_token
        # normalize
        if len(batch_windows) > 0:
            loss /= len(batch_windows)
            grad_out /= len(batch_windows)
            grad_emb /= max(1, len(batch_windows))
        # step
        self.adam_step(grad_emb, grad_out, lr=lr)
        return float(loss)


# ---------------------- Memory embedding helpers ----------------------

class MemoryDB:
    """Very small local memory DB that stores chunk-level embeddings (mean of word embeddings).
    """

    def __init__(self, path: str = '.memdb.json'):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.data = json_load(self.path)
            except Exception:
                self.data = []
        else:
            self.data = []

    def add(self, text: str, vec: np.ndarray):
        self.data.append({'text': text, 'vec': vec.tolist(), 'ts': time.time()})
        if len(self.data) > 2000:
            self.data = self.data[-2000:]

    def retrieve_vecs(self, query_vec: np.ndarray, top_k: int = 4) -> List[np.ndarray]:
        if not self.data:
            return []
        sims = []
        for entry in self.data:
            v = np.asarray(entry['vec'], dtype=np.float32)
            sims.append(float(np.dot(query_vec, v)))
        idx = np.argsort(sims)[-top_k:][::-1]
        return [np.asarray(self.data[i]['vec'], dtype=np.float32) for i in idx]

    def save(self):
        json_save(self.path, self.data)


# ---------------------- Utilities ----------------------

import json

def json_load(p: Path):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def json_save(p: Path, data):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f)


# ---------------------- Training / Run loop ----------------------

def build_corpus_from_sources(local_dir: str, urls: List[str]) -> List[str]:
    texts = []
    local_texts = load_local_texts(Path(local_dir))
    texts.extend(local_texts)
    for u in urls:
        t = fetch_text_from_url(u)
        if t and len(t) > 200:
            texts.append(t)
    return texts


def windows_from_text(text: str, tokenizer: WordTokenizer, window_size: int = 6) -> List[List[int]]:
    ids = tokenizer.encode(text).tolist()
    windows = []
    for i in range(0, max(0, len(ids) - window_size)):
        windows.append(ids[i:i + window_size])
    return windows


def train_loop(texts: List[str], model: SimpleContextModel, tokenizer: WordTokenizer, memdb: MemoryDB, epochs: int = 3, batch: int = 32, lr: float = 1e-3):
    # build windows
    windows = []
    for t in texts:
        windows.extend(windows_from_text(t, tokenizer, window_size=model.ctx + 1))
    if not windows:
        return
    # shuffle
    rng = np.random.default_rng(7)
    for epoch in range(epochs):
        rng.shuffle(windows)
        total_loss = 0.0
        for i in range(0, len(windows), batch):
            batch_windows = windows[i:i + batch]
            # for each window compute memory vectors (simple mean-of-words retrieval)
            mem_vecs_batch = []
            for w in batch_windows:
                context = w[:-1]
                ctx_vec = model.context_vector(context)
                mem_vecs = memdb.retrieve_vecs(ctx_vec, top_k=4)
                mem_vecs_batch.append(mem_vecs)
            loss = model.train_batch(batch_windows, tokenizer, memory_vectors=mem_vecs_batch, lr=lr)
            total_loss += loss * len(batch_windows)
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(windows):.4f}")


def generate_answer(prompt: str, model: SimpleContextModel, tokenizer: WordTokenizer, memdb: MemoryDB, max_words: int = 20):
    ids = tokenizer.encode(prompt).tolist()
    query_vec = model.context_vector(ids)
    mem_vecs = memdb.retrieve_vecs(query_vec, top_k=4)
    out = []
    for _ in range(max_words):
        next_id = model.predict_next(ids + [tokenizer.word_to_id.get(w, 1) for w in out], memory_vecs=mem_vecs, temperature=0.9, top_k=80, top_p=0.95)
        ids.append(next_id)
        w = tokenizer.id_to_word.get(next_id, '<UNK>')
        out.append(w)
    return ' '.join(out)


# ---------------------- CLI ----------------------

def main():
    parser = argparse.ArgumentParser(description='Improved small NumPy AI with memory and optimizer')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--urls', type=str, default='', help='File with URLs to fetch, one per line')
    parser.add_argument('--prompt', type=str, default='hello there')
    parser.add_argument('--vocab', type=int, default=4000)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--ctx', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--max-words', type=int, default=20)
    args = parser.parse_args()

    urls = []
    if args.urls:
        try:
            with open(args.urls, 'r', encoding='utf-8') as f:
                urls = [l.strip() for l in f if l.strip()]
        except Exception:
            urls = []

    print('Collecting data...')
    texts = build_corpus_from_sources(args.data_dir, urls)
    if not texts:
        print('No texts found locally or from URLs. Bootstrap sample will be used.')
        texts = [
            'hello there friend this is a small local model built with numpy for demonstration',
            'we use attention style context and local memory for retrieval',
            'training continuously on clean text yields better next-word prediction',
        ]

    print(f'Collected {len(texts)} text chunks')

    tokenizer = WordTokenizer(max_vocab=args.vocab)
    tokenizer.build(texts)
    print('Vocabulary size:', tokenizer.vocab_size)

    model = SimpleContextModel(tokenizer.vocab_size, dim=args.dim, ctx=args.ctx)
    memdb = MemoryDB(path=Path('.improved_memdb.json'))

    # populate memdb with chunk-level embeddings (mean of token embeddings)
    for t in texts:
        ids = tokenizer.encode(t)
        if len(ids) == 0:
            continue
        vec = model.context_vector(ids.tolist())
        memdb.add(t[:2000], vec)

    print('Start training...')
    train_loop(texts, model, tokenizer, memdb, epochs=args.epochs, batch=args.batch, lr=args.lr)

    print('Generating...')
    ans = generate_answer(args.prompt, model, tokenizer, memdb, max_words=args.max_words)
    print('Prompt:', args.prompt)
    print('Answer:', ans)
    memdb.save()


if __name__ == '__main__':
    main()
