#!/usr/bin/env python3
"""Word-level attention model for a Raspberry Pi Zero W.

This is a better version than the raw character model because it uses words, not
characters, and a very lightweight attention-style next-word scoring system.

It is still a small prototype, not a final "AI assistant". But it is much more
sensible than pure char-level generation on weak hardware.
"""

import argparse
import math
import re
from collections import defaultdict, Counter

import numpy as np


WORD_RE = re.compile(r"[a-zA-Z']+")


class WordTokenizer:
    def __init__(self, corpus_text=None):
        self.word_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab = ['<PAD>', '<UNK>']
        if corpus_text:
            self.fit(corpus_text)

    def fit(self, corpus_text):
        words = WORD_RE.findall(corpus_text.lower())
        counter = Counter(words)
        for word, _ in counter.most_common(250):
            if word not in self.word_to_id:
                self.word_to_id[word] = len(self.vocab)
                self.id_to_word[len(self.vocab)] = word
                self.vocab.append(word)

    def encode(self, text):
        words = WORD_RE.findall(text.lower())
        if not words:
            return np.array([1], dtype=np.int64)
        return np.array([self.word_to_id.get(w, 1) for w in words], dtype=np.int64)

    def decode(self, ids):
        words = [self.id_to_word.get(int(i), '<UNK>') for i in ids]
        return ' '.join(words)


class PiZeroWordAttention:
    def __init__(self, corpus_text, max_vocab=250, context_window=4, dim=24):
        self.tokenizer = WordTokenizer(corpus_text)
        self.max_vocab = max_vocab
        self.context_window = context_window
        self.dim = dim
        self.vocab_size = len(self.tokenizer.vocab)

        # Count-based transition matrix for local word predictions.
        self.bigram = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float32)
        self.unigram = np.zeros(self.vocab_size, dtype=np.float32)

        # Small embedding table for word similarity / attention-like scoring.
        rng = np.random.default_rng(7)
        self.emb = rng.normal(0.0, 0.2, size=(self.vocab_size, self.dim)).astype(np.float32)

        # Local training on the corpus.
        self._build_counts(corpus_text)

    def _build_counts(self, corpus_text):
        tokens = self.tokenizer.encode(corpus_text)
        if len(tokens) < 2:
            return

        for i in range(1, len(tokens)):
            prev = int(tokens[i - 1])
            curr = int(tokens[i])
            self.bigram[prev, curr] += 1.0
            self.unigram[curr] += 1.0

        # Count-based embeddings: each word gets a vector from its next-word distribution.
        for word_id in range(self.vocab_size):
            if self.unigram[word_id] <= 0:
                continue
            if self.bigram[word_id].sum() <= 0:
                continue
            probs = self.bigram[word_id] / self.bigram[word_id].sum()
            self.emb[word_id] = probs[: self.dim] if probs.size >= self.dim else np.pad(probs, (0, self.dim - probs.size), constant_values=0.0)

    def _softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        s = np.sum(e)
        return e / s

    def _attention_score(self, context_ids, candidate_id):
        if len(context_ids) == 0:
            return 0.0

        score = 0.0
        for token_id in context_ids:
            if token_id >= self.vocab_size:
                continue
            # attention-like similarity between context token and candidate word
            v1 = self.emb[token_id]
            v2 = self.emb[candidate_id]
            dot = float(np.dot(v1, v2))
            score += dot
        return score / max(1, len(context_ids))

    def predict_next(self, prompt, temperature=0.7):
        tokens = self.tokenizer.encode(prompt)
        context = list(tokens[-self.context_window:])
        if not context:
            context = [1]

        # Probability for each word candidate.
        scores = np.zeros(self.vocab_size, dtype=np.float32)
        for candidate_id in range(self.vocab_size):
            if candidate_id == 0:
                continue
            # bigram score from the most recent token
            recent = context[-1]
            base = self.bigram[recent, candidate_id]

            # attention over recent context tokens
            attn = self._attention_score(context, candidate_id)
            freq_bias = self.unigram[candidate_id]
            score = base * 4.0 + attn * 2.0 + freq_bias * 0.05
            scores[candidate_id] = score

        # avoid a silly all-zero vector
        if np.all(scores <= 0):
            scores = np.ones(self.vocab_size, dtype=np.float32)
            scores[0] = 0.0

        probs = self._softmax(scores / max(temperature, 1e-3))
        next_id = int(np.random.choice(self.vocab_size, p=probs))
        return next_id

    def generate(self, prompt, max_words=12, temperature=0.8):
        out = self.tokenizer.encode(prompt).tolist()
        for _ in range(max_words):
            next_id = self.predict_next(self.tokenizer.decode(out), temperature=temperature)
            if next_id == 0:
                break
            out.append(next_id)
            word = self.tokenizer.id_to_word.get(next_id, '<UNK>')
            if word in {'.', '!', '?'}:
                continue
        return self.tokenizer.decode(out)


def build_demo_corpus():
    return """
    hello there friend how are you today
    hello there friend i am good and ready to help
    good to see you again and hello there friend
    today is a good day for learning and building models
    we use attention to look at recent words and choose the next one
    attention helps the model focus on important context and memory
    this is a tiny word model for a raspberry pi zero
    the model is small but the idea is clear and useful
    """


def main():
    parser = argparse.ArgumentParser(description='Tiny word-level transformer-style model for Pi Zero W.')
    parser.add_argument('--prompt', type=str, default='hello there')
    parser.add_argument('--max_words', type=int, default=10)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--data', type=str, default='')
    args = parser.parse_args()

    if args.data:
        with open(args.data, 'r', encoding='utf-8', errors='ignore') as f:
            corpus = f.read()
    else:
        corpus = build_demo_corpus()

    model = PiZeroWordAttention(corpus)
    result = model.generate(args.prompt, max_words=args.max_words, temperature=args.temperature)
    print('Prompt:', args.prompt)
    print('Generated:', result)


if __name__ == '__main__':
    main()
