#!/usr/bin/env python3
"""Build a larger local training corpus from text files in a folder.

This is the honest way to get more data.
It does NOT create fake billions of tokens; it reads real text files from your
local project and combines them into one training corpus for a small word model.
"""

import argparse
import re
from pathlib import Path


def iter_text_files(root: Path):
    exts = {'.txt', '.md', '.json', '.csv', '.log', '.py', '.js', '.ts', '.html', '.xml'}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def clean_text(text: str) -> str:
    text = text.replace('\r', '\n')
    text = text.replace('\t', ' ')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description='Build a larger training corpus from local text files.')
    parser.add_argument('--input', type=str, default='data', help='Folder to scan for text files')
    parser.add_argument('--output', type=str, default='data/corpus.txt', help='Output file')
    parser.add_argument('--min_chars', type=int, default=10, help='Ignore tiny files below this size')
    args = parser.parse_args()

    root = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    chunks = []
    total = 0
    count = 0
    for p in iter_text_files(root):
        try:
            txt = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        txt = clean_text(txt)
        if len(txt) < args.min_chars:
            continue
        chunks.append(txt)
        total += len(txt)
        count += 1

    if not chunks:
        raise FileNotFoundError(f'No text files found under: {root}')

    corpus = '\n\n'.join(chunks)
    out.write_text(corpus, encoding='utf-8')
    print(f'Wrote corpus to {out}')
    print(f'Files used: {count}')
    print(f'Characters: {total}')


if __name__ == '__main__':
    main()
