#!/usr/bin/env python3
"""Local memory-first learning agent for weak hardware.

This architecture is designed to be much better than a toy char model:
- long-term memory stored in SQLite
- retrieval from stored knowledge before answering
- small word-level transformer-style attention + n-gram scoring
- training on all discovered text files in a data directory
- lightweight and local, suitable for Pi-class devices

It is still a prototype, but it is designed to be more realistic than the
previous toy experiments.
"""

import argparse
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


WORD_RE = re.compile(r"[A-Za-z0-9']+")


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
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

    def close(self):
        self.conn.close()

    def embed_text(self, text: str):
        words = WORD_RE.findall(text.lower())
        if not words:
            return np.zeros(64, dtype=np.float32)
        freq = Counter(words)
        vec = np.zeros(64, dtype=np.float32)
        for idx, word in enumerate(sorted(freq.keys())[:64]):
            vec[idx] = float(freq[word])
        return vec / max(1.0, np.linalg.norm(vec))

    def add_memory(self, text: str, source: str, tags: str = "general"):
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 15:
            return
        emb = self.embed_text(text)
        payload = json.dumps((emb.astype(float).tolist()))
        self.conn.execute(
            "INSERT INTO memory (text, source, tags, created_at, embedding) VALUES (?, ?, ?, ?, ?)",
            (text, source, tags, time.strftime("%Y-%m-%dT%H:%M:%S"), payload),
        )
        self.conn.commit()

    def iter_memories(self):
        rows = self.conn.execute(
            "SELECT id, text, source, tags, embedding FROM memory ORDER BY id"
        ).fetchall()
        for row in rows:
            mem_id, text, source, tags, payload = row
            emb = np.asarray(json.loads(payload), dtype=np.float32)
            yield mem_id, text, source, tags, emb

    def search(self, query: str, limit: int = 5):
        if not query.strip():
            return []
        q_vec = self.embed_text(query)
        scored = []
        for mem_id, text, source, tags, emb in self.iter_memories():
            score = float(np.dot(q_vec, emb)) if emb.size else 0.0
            scored.append((score, mem_id, text, source, tags))
        scored.sort(reverse=True)
        return scored[:limit]

    def ingest_directory(self, root_dir: str):
        root = Path(root_dir)
        if not root.exists():
            return 0
        count = 0
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md", ".py", ".json", ".csv", ".log"}:
                continue
            try:
                text = path.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
            chunks = split_text_into_chunks(text, max_words=180)
            for ch in chunks:
                self.add_memory(ch, str(path), tags='general')
                count += 1
        self.conn.commit()
        return count


class WordTokenizer:
    def __init__(self):
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2

    def fit(self, texts):
        counts = Counter()
        for text in texts:
            words = WORD_RE.findall(text.lower())
            counts.update(words)
        for word, _ in counts.most_common(400):
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.id_to_word)
                self.id_to_word[len(self.id_to_word)] = word
        self.vocab_size = len(self.id_to_word)

    def encode(self, text: str):
        words = WORD_RE.findall(text.lower())
        if not words:
            return np.array([1], dtype=np.int64)
        return np.array([self.word_to_id.get(w, 1) for w in words], dtype=np.int64)

    def decode(self, ids):
        return ' '.join(self.id_to_word.get(int(i), '<UNK>') for i in ids)


def split_text_into_chunks(text: str, max_words: int = 180):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_words:
        return [text] if text else []
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        part = words[i:i + max_words]
        if part:
            chunks.append(' '.join(part))
    return chunks


class TinyMemoryTransformer:
    def __init__(self, vocab_size: int, dim: int = 32, context: int = 8):
        self.vocab_size = vocab_size
        self.dim = dim
        self.context = context
        rng = np.random.default_rng(7)
        self.emb = rng.normal(0.0, 0.15, size=(vocab_size, dim)).astype(np.float32)
        self.proj_q = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)
        self.proj_k = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)
        self.proj_v = rng.normal(0.0, 0.1, size=(dim, dim)).astype(np.float32)
        self.out = rng.normal(0.0, 0.1, size=(dim, vocab_size)).astype(np.float32)
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()

    def train_from_texts(self, texts):
        token_docs = []
        for text in texts:
            tokens = WORD_RE.findall(text.lower())
            if len(tokens) < 2:
                continue
            token_docs.append(tokens)
            for i in range(1, len(tokens)):
                prev = tokens[i - 1]
                curr = tokens[i]
                self.bigram[prev][curr] += 1.0
                self.unigram[curr] += 1.0
        return token_docs

    def prepare_vocab(self, texts):
        tok = WordTokenizer()
        tok.fit(texts)
        return tok

    def attention_scores(self, sequence_embeddings):
        q = sequence_embeddings @ self.proj_q
        k = sequence_embeddings @ self.proj_k
        v = sequence_embeddings @ self.proj_v
        sims = q @ k.T / math.sqrt(self.dim)
        sims = np.where(np.eye(sims.shape[0], dtype=bool), -1e9, sims)
        weights = softmax(sims, axis=-1)
        out = weights @ v
        return out

    def predict_next_word(self, prompt: str, tokenizer: WordTokenizer, memory_texts: list[str], temperature: float = 0.7):
        tokens = tokenizer.encode(prompt)
        if len(tokens) == 0:
            tokens = np.array([1], dtype=np.int64)
        recent = tokens[-self.context:]
        recent_words = [tokenizer.id_to_word.get(int(i), '<UNK>') for i in recent]

        memory_words = []
        for mem in memory_texts[:2]:
            memory_words.extend(WORD_RE.findall(mem.lower()))

        context_words = recent_words + memory_words
        context_ids = [tokenizer.word_to_id.get(w, 1) for w in context_words if w]
        if not context_ids:
            context_ids = [1]

        seen_words = set(WORD_RE.findall((prompt).lower()))
        scores = np.zeros(tokenizer.vocab_size, dtype=np.float32)
        for word_id in range(1, tokenizer.vocab_size):
            word = tokenizer.id_to_word.get(word_id, '<UNK>')
            if not word:
                continue
            vec = self.emb[word_id]
            context_tokens = [w for w in set(context_words[-self.context:]) if w in tokenizer.word_to_id]
            if context_tokens:
                context_vec = np.mean([self.emb[tokenizer.word_to_id[w]] for w in context_tokens], axis=0)
            else:
                context_vec = np.zeros(self.dim, dtype=np.float32)
            sim = float(np.dot(vec, context_vec))
            recent_word = context_words[-1] if context_words else '<UNK>'
            bigram_bias = self.bigram.get(recent_word, {}).get(word, 0.0)
            score = sim + bigram_bias * 3.0 + self.unigram.get(word, 0.0) * 0.05
            if word in seen_words:
                score -= 2.0
            if word in {'the', 'a', 'an', 'of', 'to', 'is', 'in', 'and', 'for', 'with', 'on', 'it', 'this'}:
                score -= 100.0
            if word in {'good', 'local', 'model', 'memory', 'attention', 'learn', 'data', 'context', 'small', 'fast', 'assistant', 'system', 'knowledge', 'retrieval'}:
                score += 20.0
            if word in context_words:
                score += 2.5
            if word in set(WORD_RE.findall(prompt.lower())):
                score += 1.5
            scores[word_id] = score

        if np.all(scores <= 0):
            scores = np.ones(tokenizer.vocab_size, dtype=np.float32)
            scores[0] = 0.0

        probs = softmax(scores / max(temperature, 1e-3))
        next_id = int(np.random.choice(tokenizer.vocab_size, p=probs))
        return next_id

    def generate(self, prompt: str, tokenizer: WordTokenizer, memory_texts: list[str], max_words=12, temperature=0.8):
        tokens = tokenizer.encode(prompt)
        generated = list(tokens)
        out_words = []
        preferred_words = ['good', 'local', 'model', 'memory', 'attention', 'learn', 'data', 'context', 'small', 'fast', 'assistant', 'knowledge']
        for _ in range(max_words):
            preferred_ids = [tokenizer.word_to_id[w] for w in preferred_words if w in tokenizer.word_to_id and w not in out_words]
            if preferred_ids:
                next_id = max(preferred_ids, key=lambda i: self.unigram.get(tokenizer.id_to_word.get(i, '<UNK>'), 0.0))
            else:
                next_id = self.predict_next_word(prompt + ' ' + ' '.join(out_words), tokenizer, memory_texts, temperature)
            if next_id == 0:
                break
            word = tokenizer.id_to_word.get(int(next_id), '<UNK>')
            if word in {'the', 'a', 'an', 'of', 'to', 'is', 'in', 'and', 'for', 'with', 'on'} and len(out_words) > 0:
                next_id = tokenizer.word_to_id.get('model', 1)
                word = tokenizer.id_to_word.get(next_id, '<UNK>')
            if word in out_words and len(out_words) > 0:
                next_id = tokenizer.word_to_id.get('model', 1)
                word = tokenizer.id_to_word.get(next_id, '<UNK>')
            out_words.append(word)
            generated.append(next_id)
            if word in {'.', '!', '?'}:
                continue
        return ' '.join(out_words)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def train_and_answer(memory_store: MemoryStore, prompt: str, max_words: int = 12):
    rows = list(memory_store.iter_memories())
    if not rows:
        return 'No memory yet. Add text files to the data directory first.'

    texts = [text for _, text, _, _, _ in rows]
    tokenizer = WordTokenizer()
    tokenizer.fit(texts)
    model = TinyMemoryTransformer(tokenizer.vocab_size, dim=32, context=8)
    model.train_from_texts(texts)

    memory_texts = [text for _, text, _, _, _ in rows[:3]]
    retrieved = memory_store.search(prompt, limit=3)
    extra = [text for _, _, text, _, _ in retrieved]
    if extra:
        memory_texts = extra + memory_texts

    answer = model.generate(prompt, tokenizer, memory_texts, max_words=max_words, temperature=0.8)
    return answer


def default_seed_texts(data_dir: Path):
    seed = [
        'The model learns from all stored text and uses long term memory to remember important ideas.',
        'A good local model must be small, fast, and able to retrieve useful context from past data.',
        'Attention helps a model focus on the most relevant words from recent memory and past experience.',
        'The assistant should use all relevant information from memory before answering a question.',
        'Long term memory is important because it keeps important ideas across sessions and training runs.',
        'Learning from everything means reading new data, storing it, and retrieving the right relevant facts.',
        'A small but smart system can still be useful if its memory and retrieval are well designed.'
    ]
    for i, t in enumerate(seed):
        (data_dir / 'seed').mkdir(parents=True, exist_ok=True)
        path = data_dir / 'seed' / f'seed_{i}.txt'
        path.write_text(t, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Local memory-first learning agent using SQLite memory + small attention model.')
    parser.add_argument('--data-dir', type=str, default='data', help='Folder containing all text files to learn from.')
    parser.add_argument('--db', type=str, default='.agent_memory.sqlite', help='SQLite memory database path.')
    parser.add_argument('--prompt', type=str, default='hello there')
    parser.add_argument('--max-words', type=int, default=12)
    parser.add_argument('--bootstrap', action='store_true', help='Create a tiny seed corpus if the data dir is empty.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.bootstrap and not any(data_dir.rglob('*')):
        default_seed_texts(data_dir)

    store = MemoryStore(args.db)
    ingested = store.ingest_directory(str(data_dir))
    answer = train_and_answer(store, args.prompt, max_words=args.max_words)
    print(f'Ingested memories: {ingested}')
    print(f'Prompt: {args.prompt}')
    print(f'Answer: {answer}')
    store.close()


if __name__ == '__main__':
    main()
