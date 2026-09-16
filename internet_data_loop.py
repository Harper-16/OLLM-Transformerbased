#!/usr/bin/env python3
"""Download a small stream of text from the web and keep training a tiny word model.

This is a loop for a very small local model. It is still a toy, but it matches
what you asked: fetch data, save it under data/, build a corpus, and retrain.

It does not magically make a Pi Zero W into a real assistant. It does give you
an honest, repeatable local data pipeline that keeps improving the tiny model
as more text arrives.
"""

import argparse
import re
import time
from pathlib import Path
from urllib import request, error
from html.parser import HTMLParser

from pi_zero_word_transformer import PiZeroWordAttention


DATA_DIR = Path(__file__).resolve().parent / 'data'
RAW_DIR = DATA_DIR / 'raw'


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'svg', 'noscript'}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {'script', 'style', 'svg', 'noscript'}:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            text = re.sub(r'\s+', ' ', data)
            if text.strip():
                self.chunks.append(text.strip())

    def get_text(self):
        return ' '.join(self.chunks)


def fetch_url_text(url: str, timeout: int = 15) -> str:
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    text = body.decode('utf-8', errors='ignore')
    parser = TextExtractor()
    parser.feed(text)
    out = parser.get_text()
    return re.sub(r'\s+', ' ', out)


def save_text_file(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', errors='ignore') as f:
        f.write(text[:200000])


def build_corpus_from_dir(data_dir: Path, output_path: Path):
    files = sorted(data_dir.rglob('*.txt'))
    if not files:
        return False

    chunks = []
    for f in files:
        try:
            txt = f.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        txt = re.sub(r'\s+', ' ', txt)
        if len(txt.strip()) > 40:
            chunks.append(txt.strip())

    if not chunks:
        return False

    out = '\n\n'.join(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(out, encoding='utf-8')
    return True


def get_sources():
    return [
        'https://en.wikipedia.org/wiki/Artificial_intelligence',
        'https://en.wikipedia.org/wiki/Computer_science',
        'https://en.wikipedia.org/wiki/Raspberry_Pi',
        'https://en.wikipedia.org/wiki/Transformers_(machine_learning_model)',
        'https://en.wikipedia.org/wiki/Attention_(machine_learning)',
        'https://www.gutenberg.org/ebooks/100',
        'https://www.gutenberg.org/ebooks/11',
        'https://www.gutenberg.org/ebooks/76',
        'https://www.gutenberg.org/ebooks/84',
    ]


def download_round(data_dir: Path):
    raw_dir = data_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    urls = get_sources()
    saved = 0

    for idx, url in enumerate(urls):
        try:
            text = fetch_url_text(url, timeout=20)
        except Exception:
            continue
        if len(text) < 200:
            continue
        target = raw_dir / f'web_{idx}.txt'
        save_text_file(target, text)
        saved += 1

    return saved


def train_round(data_dir: Path, prompt: str, steps: int = 40, max_words: int = 10):
    corpus = data_dir / 'corpus.txt'
    if not build_corpus_from_dir(data_dir / 'raw', corpus):
        return f'No raw text to train on yet. Downloaded {len(list((data_dir / "raw").glob("*.txt")))} files.'

    text = corpus.read_text(encoding='utf-8', errors='ignore')
    model = PiZeroWordAttention(text)
    out = model.generate(prompt, max_words=max_words, temperature=0.8)
    return out


def main():
    parser = argparse.ArgumentParser(description='Download small public text sources and continuously train the tiny word model.')
    parser.add_argument('--prompt', type=str, default='hello there')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--max_words', type=int, default=10)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--sleep', type=float, default=0.0)
    parser.add_argument('--loop', action='store_true', help='Repeat forever until Ctrl-C')
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    round_count = 0
    while True:
        saved = download_round(DATA_DIR)
        result = train_round(DATA_DIR, prompt=args.prompt, steps=args.steps, max_words=args.max_words)
        print(f'Round {round_count + 1}: downloaded={saved} | output={result}')
        round_count += 1

        if not args.loop:
            if round_count >= args.rounds:
                break
        if args.sleep > 0:
            time.sleep(args.sleep)

        if args.loop is False and round_count >= args.rounds:
            break


if __name__ == '__main__':
    main()
