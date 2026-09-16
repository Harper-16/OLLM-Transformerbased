#!/usr/bin/env python3
"""
Data preparation pipeline for LLM training.

Steps included:
 - normalize: HTML strip, unicode normalization, whitespace collapse
 - dedupe_exact: exact dedupe by SHA256
 - dedupe_minhash: MinHash near-duplicate removal
 - filter_lang: language detection (langdetect)
 - filter_pii: basic PII detection (regex) - redact or drop
 - split: create train/val/test splits
 - train_tokenizer: train SentencePiece tokenizer
 - tokenize_shard: tokenize with SentencePiece and shard into JSONL.gz shards
 - run_all: run a recommended sequence of steps

Input/Output: newline-delimited JSON (JSONL). Each item expects at least 'text' and optional metadata fields (source, license).

This script is intended as a practical, readable starting point. For very large datasets, run per-step on machines with appropriate resources and/or adapt to streaming/distributed processing.
"""

import argparse
import os
import sys
import json
import hashlib
import unicodedata
import re
import gzip
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Iterable, List
from langdetect import detect_langs
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import sentencepiece as spm
import random
from pathlib import Path

# --------------------- I/O helpers ---------------------

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl_gzip(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_urlopen(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode('utf-8', errors='replace')

# --------------------- Downloaders ---------------------

def canonical_hf_name(dataset_name):
    aliases = {
        'c4': 'allenai/c4',
        'imdb': 'stanfordnlp/imdb',
        'wikitext': 'Salesforce/wikitext',
        'wikipedia': 'wikipedia',
        'openwebtext': 'openwebtext',
        'pile': 'EleutherAI/the_pile',
        'the_pile': 'EleutherAI/the_pile',
        'books': 'bookcorpusopen/bookcorpusopen',
        'bookcorpus': 'bookcorpusopen/bookcorpusopen',
        'common_voice': 'mozilla-foundation/common_voice_17_0',
        'cnn_dailymail': 'abisee/cnn_dailymail',
    }
    if '/' in dataset_name:
        return dataset_name
    return aliases.get(dataset_name.lower(), dataset_name)


def download_hf_dataset(dataset_name, output_path, split='train', text_field='text', limit=None, config=None, streaming=False):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError('datasets package is required for Hugging Face downloads. Install with pip install -r data_pipeline/requirements.txt') from exc

    dataset_name = canonical_hf_name(dataset_name)
    dataset_kwargs = {}
    if config is not None:
        dataset_kwargs['name'] = config
    ds = load_dataset(dataset_name, split=split, **dataset_kwargs)
    if streaming:
        # stream mode isn't fully supported by this script; fallback to non-streaming
        streaming = False
    if limit is not None:
        limit = min(limit, len(ds))
        ds = ds.select(range(limit))

    with open(output_path, 'w', encoding='utf-8') as f:
        for row in ds:
            text = row.get(text_field)
            if not text:
                continue
            rec = {'text': str(text), 'source': dataset_name, 'license': 'hf-dataset'}
            for extra_key in ('title', 'url', 'id', 'context', 'question', 'answer'):
                if extra_key in row and row[extra_key] is not None:
                    rec[extra_key] = row[extra_key]
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Wrote {len(ds)} rows from Hugging Face dataset {dataset_name} to {output_path}')


def download_wikipedia(query, output_path, limit=20, language='en'):
    search_url = (
        'https://{lang}.wikipedia.org/w/api.php?'
        'action=query&format=json&list=search&srsearch={q}&srlimit={limit}&utf8=1'
    ).format(lang=language, q=urllib.parse.quote_plus(query), limit=limit)
    search_json = json.loads(safe_urlopen(search_url))
    results = search_json.get('query', {}).get('search', [])
    if not results:
        print(f'No Wikipedia results for query: {query}')
        return

    out = []
    for item in results:
        pageid = item.get('pageid')
        title = item.get('title')
        extract_url = (
            'https://{lang}.wikipedia.org/w/api.php?'
            'action=query&format=json&prop=extracts&explaintext=1&pageids={pageid}&utf8=1'
        ).format(lang=language, pageid=pageid)
        try:
            extract_json = json.loads(safe_urlopen(extract_url))
        except Exception:
            continue
        pages = extract_json.get('query', {}).get('pages', {})
        page = next(iter(pages.values()), {})
        text = page.get('extract')
        if not text:
            continue
        out.append({'text': text.strip(), 'source': 'wikipedia', 'title': title, 'language': language, 'license': 'wikimedia'})
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Wrote {len(out)} Wikipedia pages to {output_path}')


def download_arxiv(query, output_path, limit=20):
    url = 'https://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={limit}'.format(
        q=urllib.parse.quote_plus(query),
        limit=limit,
    )
    xml_text = safe_urlopen(url)
    root = ET.fromstring(xml_text)
    atom_ns = '{http://www.w3.org/2005/Atom}'
    entries = root.findall(f'{atom_ns}entry')
    out = []
    for entry in entries:
        title = ''.join(entry.find(f'{atom_ns}title').itertext()).replace('\n', ' ').strip()
        summary = ''.join(entry.find(f'{atom_ns}summary').itertext()).replace('\n', ' ').strip()
        text = f'{title}\n{summary}'
        if text:
            out.append({'text': text, 'source': 'arxiv', 'title': title, 'license': 'arxiv-api'})
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Wrote {len(out)} arXiv entries to {output_path}')


def download_url_list(url_list_path, output_path, per_page_limit=50):
    out = []
    with open(url_list_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    for idx, url in enumerate(urls[:per_page_limit]):
        try:
            html = safe_urlopen(url)
            text = re.sub(r'<script.*?</script>', ' ', html, flags=re.S | re.I)
            text = re.sub(r'<style.*?</style>', ' ', text, flags=re.S | re.I)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            if text:
                out.append({'text': text, 'source': url, 'license': 'url-list'})
        except Exception:
            continue
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Wrote {len(out)} webpages from url list to {output_path}')


def download_sources(sources, out_dir, limit_per_source=100, dataset_split='train', hf_config=None, wikipedia_query='artificial intelligence', arxiv_query='artificial intelligence', url_list=None, language='en'):
    os.makedirs(out_dir, exist_ok=True)
    combined_path = os.path.join(out_dir, 'raw_combined.jsonl')
    if os.path.exists(combined_path):
        os.remove(combined_path)
    for source in sources:
        source = source.strip().lower()
        target = os.path.join(out_dir, f'{source}.jsonl')
        if source == 'hf':
            dataset_name = hf_config or 'c4'
            download_hf_dataset(dataset_name, target, split=dataset_split, text_field='text', limit=limit_per_source, config=None)
        elif source == 'wikipedia':
            download_wikipedia(wikipedia_query, target, limit=limit_per_source, language=language)
        elif source == 'arxiv':
            download_arxiv(arxiv_query, target, limit=limit_per_source)
        elif source == 'urls':
            if url_list is None:
                raise ValueError('A URL list file is required when source is "urls"')
            download_url_list(url_list, target, per_page_limit=limit_per_source)
        else:
            raise ValueError(f'Unknown source {source!r}. Supported sources: hf, wikipedia, arxiv, urls')
        # Merge into one combined raw file
        with open(target, 'r', encoding='utf-8') as f_in, open(combined_path, 'a', encoding='utf-8') as f_out:
            for line in f_in:
                if line.strip():
                    f_out.write(line)
    print(f'Combined downloads saved to {combined_path}')
    return combined_path

# --------------------- Normalization ---------------------

HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTISPACE_RE = re.compile(r"\s+")


def normalize_text(text):
    if not text:
        return ""
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # Strip HTML-ish tags
    text = HTML_TAG_RE.sub(' ', text)
    # Remove control chars
    text = ''.join(ch for ch in text if ch == '\n' or (31 < ord(ch) < 0x110000))
    # Collapse whitespace
    text = MULTISPACE_RE.sub(' ', text)
    text = text.strip()
    return text

# --------------------- Exact dedupe ---------------------

def sha256_of_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def dedupe_exact(input_path, output_path):
    seen = set()
    out_items = []
    for doc in tqdm(read_jsonl(input_path), desc='Exact dedupe'):
        text = doc.get('text', '')
        norm = normalize_text(text)
        if not norm:
            continue
        h = sha256_of_text(norm)
        if h in seen:
            continue
        seen.add(h)
        doc['text'] = norm
        doc['hash'] = h
        out_items.append(doc)
    write_jsonl(output_path, out_items)
    print(f'Wrote {len(out_items)} items to {output_path}')

# --------------------- MinHash near-duplicate removal ---------------------

def shingle(text, k=5):
    # character shingles
    text = text.replace(' ', '_')
    return [text[i:i+k] for i in range(max(1, len(text) - k + 1))]


def dedupe_minhash(input_path, output_path, threshold=0.9, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    kept = []
    idx = 0
    for doc in tqdm(read_jsonl(input_path), desc='MinHash dedupe - building'):  # expects normalized and exact-deduped input
        text = doc.get('text', '')
        if not text:
            continue
        shingles = shingle(text, k=5)
        m = MinHash(num_perm=num_perm)
        for s in shingles:
            m.update(s.encode('utf-8'))
        # query existing
        res = lsh.query(m)
        if res:
            # similar doc(s) already present; skip
            continue
        key = f'doc{idx}'
        lsh.insert(key, m)
        minhashes[key] = m
        kept.append(doc)
        idx += 1
    write_jsonl(output_path, kept)
    print(f'Wrote {len(kept)} (minhash-filtered) items to {output_path}')

# --------------------- Language filtering ---------------------

def filter_lang(input_path, output_path, target_lang='en', prob_threshold=0.8):
    kept = []
    for doc in tqdm(read_jsonl(input_path), desc='Language filtering'):
        text = doc.get('text', '')
        if not text:
            continue
        try:
            langs = detect_langs(text)
            if not langs:
                continue
            top = langs[0]
            lang = top.lang
            prob = float(top.prob)
            doc['lang_detect'] = f'{lang}:{prob:.3f}'
            if lang == target_lang and prob >= prob_threshold:
                kept.append(doc)
        except Exception:
            continue
    write_jsonl(output_path, kept)
    print(f'Wrote {len(kept)} lang-{target_lang} items to {output_path}')

# --------------------- Basic PII detection & redaction ---------------------

EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_RE = re.compile(r"\b(?:\+\d{1,3}[- ]?)?(?:\(\d{1,4}\)|\d{1,4})[- .]?\d{1,4}[- .]?\d{1,9}\b")


def filter_or_redact_pii(input_path, output_path, action='redact'):
    # action: 'redact' or 'drop'
    out = []
    for doc in tqdm(read_jsonl(input_path), desc='PII filtering'):
        text = doc.get('text', '')
        if not text:
            continue
        has = False
        if EMAIL_RE.search(text) or SSN_RE.search(text) or PHONE_RE.search(text):
            has = True
        if has:
            if action == 'drop':
                continue
            else:
                text = EMAIL_RE.sub('[REDACTED_EMAIL]', text)
                text = SSN_RE.sub('[REDACTED_SSN]', text)
                text = PHONE_RE.sub('[REDACTED_PHONE]', text)
                doc['text'] = text
                doc.setdefault('tags', []).append('pii_redacted')
        out.append(doc)
    write_jsonl(output_path, out)
    print(f'Wrote {len(out)} items after PII {action}')

# --------------------- Splitting ---------------------

def split_dataset(input_path, out_dir, train_frac=0.98, val_frac=0.01, test_frac=0.01, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    items = list(read_jsonl(input_path))
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, 'train.jsonl'), train)
    write_jsonl(os.path.join(out_dir, 'validation.jsonl'), val)
    write_jsonl(os.path.join(out_dir, 'test.jsonl'), test)
    print(f'Splits: train={len(train)}, val={len(val)}, test={len(test)} saved to {out_dir}')

# --------------------- Tokenizer training (SentencePiece) ---------------------

def cap_vocab_size_for_corpus(input_path, vocab_size, min_allowed=128):
    # SentencePiece enforces vocab_size <= actual learned vocabulary size.
    # For tiny corpora, the learned vocabulary is much smaller than the requested size.
    # We cap it to a sane, data-aware bound so small test runs do not crash.
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return max(min_allowed, vocab_size)
    # bound the requested vocab to a reasonable multiple of corpus size, but never below a floor
    corpus_bound = max(min_allowed, min(vocab_size, len(lines) * 50))
    return min(vocab_size, corpus_bound)


def train_sentencepiece(input_path, model_prefix, vocab_size=50000, model_type='unigram'):
    # input_path is a text file with one document per line (large)
    # SentencePiece training flags
    adjusted_vocab_size = cap_vocab_size_for_corpus(input_path, vocab_size)
    if adjusted_vocab_size != vocab_size:
        print(f'Warning: original vocab_size={vocab_size} is too high for the corpus; reducing to {adjusted_vocab_size}.')
    spm_args = (
        f"--input={input_path} --model_prefix={model_prefix} --vocab_size={adjusted_vocab_size}"
        f" --model_type={model_type} --character_coverage=0.9995 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3"
    )
    print('Training sentencepiece with args:', spm_args)
    spm.SentencePieceTrainer.Train(spm_args)
    print('Trained tokenizer:', model_prefix + '.model')

def export_texts_for_tokenizer(input_jsonl, out_txt):
    # Writes one normalized text per line for tokenizer training
    with open(out_txt, 'w', encoding='utf-8') as fw:
        for doc in read_jsonl(input_jsonl):
            t = doc.get('text', '').replace('\n', ' ')
            if t:
                fw.write(t + '\n')
    print('Wrote texts for tokenizer to', out_txt)

# --------------------- Tokenize & Shard ---------------------

def tokenize_and_shard(input_jsonl, sp_model_path, out_dir, samples_per_shard=5000):
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    os.makedirs(out_dir, exist_ok=True)
    shard_idx = 0
    out_items = []
    count = 0
    for doc in tqdm(read_jsonl(input_jsonl), desc='Tokenize & shard'):
        text = doc.get('text', '')
        if not text:
            continue
        ids = sp.EncodeAsIds(text)
        doc_out = {
            'ids': ids,
            'text': text,
            'hash': doc.get('hash'),
            'source': doc.get('source'),
            'license': doc.get('license'),
            'tags': doc.get('tags', []),
        }
        out_items.append(doc_out)
        count += 1
        if count >= samples_per_shard:
            shard_path = os.path.join(out_dir, f'shard_{shard_idx:06d}.jsonl.gz')
            write_jsonl_gzip(shard_path, out_items)
            print(f'Wrote shard {shard_path} containing {len(out_items)} samples')
            shard_idx += 1
            out_items = []
            count = 0
    # final
    if out_items:
        shard_path = os.path.join(out_dir, f'shard_{shard_idx:06d}.jsonl.gz')
        write_jsonl_gzip(shard_path, out_items)
        print(f'Wrote shard {shard_path} containing {len(out_items)} samples')

# --------------------- Orchestration CLI ---------------------

def run_all_pipeline(input, out_dir, tmp_dir, lang='en', samples_per_shard=5000, pii_action='redact'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    n1 = os.path.join(tmp_dir, 'normalized.jsonl')
    dedup1 = os.path.join(tmp_dir, 'dedup_exact.jsonl')
    dedup2 = os.path.join(tmp_dir, 'dedup_minhash.jsonl')
    langf = os.path.join(tmp_dir, 'lang.jsonl')
    piif = os.path.join(tmp_dir, 'pii.jsonl')
    split_dir = os.path.join(out_dir, 'splits')
    tokenizer_text = os.path.join(tmp_dir, 'for_tokenizer.txt')
    model_prefix = os.path.join(out_dir, 'tokenizer')
    token_model = model_prefix + '.model'
    tokenized_out = os.path.join(out_dir, 'shards')

    print('Step 1: Exact dedupe (normalization + exact dedupe)')
    # Normalize & exact dedupe
    # normalize is done inside dedupe_exact via normalize_text
    dedupe_exact(input, dedup1)

    print('Step 2: Near-duplicate removal (MinHash)')
    dedupe_minhash(dedup1, dedup2)

    print('Step 3: Language filter')
    filter_lang(dedup2, langf, target_lang=lang)

    print('Step 4: PII redaction/drop')
    filter_or_redact_pii(langf, piif, action=pii_action)

    print('Step 5: Splitting')
    split_dataset(piif, split_dir)

    print('Step 6: Train tokenizer on train split')
    export_texts_for_tokenizer(os.path.join(split_dir, 'train.jsonl'), tokenizer_text)
    train_sentencepiece(tokenizer_text, model_prefix)

    print('Step 7: Tokenize & shard train+val+test into', tokenized_out)
    # Combine splits into one file to tokenize & shard
    combined = os.path.join(tmp_dir, 'combined.jsonl')
    items = []
    for path in [os.path.join(split_dir, 'train.jsonl'), os.path.join(split_dir, 'validation.jsonl'), os.path.join(split_dir, 'test.jsonl')]:
        for d in read_jsonl(path):
            items.append(d)
    write_jsonl(combined, items)
    tokenize_and_shard(combined, token_model, tokenized_out, samples_per_shard=samples_per_shard)

    print('Pipeline complete. Outputs in', out_dir)


def main():
    p = argparse.ArgumentParser(description='Data pipeline for LLM training')
    sub = p.add_subparsers(dest='cmd')

    # normalize (deprecated: normalization integrated into dedupe_exact)
    a = sub.add_parser('normalize')
    a.add_argument('--input', required=True)
    a.add_argument('--output', required=True)

    a = sub.add_parser('dedupe_exact')
    a.add_argument('--input', required=True)
    a.add_argument('--output', required=True)

    a = sub.add_parser('dedupe_minhash')
    a.add_argument('--input', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--threshold', type=float, default=0.9)
    a.add_argument('--num_perm', type=int, default=128)

    a = sub.add_parser('filter_lang')
    a.add_argument('--input', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--lang', default='en')
    a.add_argument('--prob', type=float, default=0.8)

    a = sub.add_parser('filter_pii')
    a.add_argument('--input', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--action', choices=['redact', 'drop'], default='redact')

    a = sub.add_parser('split')
    a.add_argument('--input', required=True)
    a.add_argument('--out_dir', required=True)
    a.add_argument('--train_frac', type=float, default=0.98)
    a.add_argument('--val_frac', type=float, default=0.01)
    a.add_argument('--test_frac', type=float, default=0.01)
    a.add_argument('--seed', type=int, default=42)

    a = sub.add_parser('train_tokenizer')
    a.add_argument('--input', required=True, help='Text file (one document per line)')
    a.add_argument('--model_prefix', required=True)
    a.add_argument('--vocab_size', type=int, default=50000)
    a.add_argument('--model_type', choices=['unigram', 'bpe', 'char', 'word'], default='unigram')

    a = sub.add_parser('tokenize_shard')
    a.add_argument('--input', required=True, help='Input JSONL to tokenize')
    a.add_argument('--sp_model', required=True, help='SentencePiece model .model file')
    a.add_argument('--out_dir', required=True)
    a.add_argument('--samples_per_shard', type=int, default=5000)

    a = sub.add_parser('download')
    a.add_argument('--source', choices=['hf', 'wikipedia', 'arxiv', 'urls'], required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--dataset', default='c4', help='Hugging Face dataset name for source=hf')
    a.add_argument('--split', default='train')
    a.add_argument('--text_field', default='text')
    a.add_argument('--limit', type=int, default=100)
    a.add_argument('--config', default=None)
    a.add_argument('--query', default='artificial intelligence')
    a.add_argument('--lang', default='en')
    a.add_argument('--url_list', default=None, help='Path to a text file with one URL per line for source=urls')

    a = sub.add_parser('download_all')
    a.add_argument('--out_dir', required=True)
    a.add_argument('--limit_per_source', type=int, default=100)
    a.add_argument('--sources', default='hf,wikipedia,arxiv', help='Comma-separated list of sources: hf,wikipedia,arxiv,urls')
    a.add_argument('--dataset', default='c4')
    a.add_argument('--split', default='train')
    a.add_argument('--query', default='artificial intelligence')
    a.add_argument('--lang', default='en')
    a.add_argument('--url_list', default=None)

    a = sub.add_parser('run_all')
    a.add_argument('--input', required=True)
    a.add_argument('--out_dir', required=True)
    a.add_argument('--tmp_dir', required=True)
    a.add_argument('--lang', default='en')
    a.add_argument('--samples_per_shard', type=int, default=5000)
    a.add_argument('--pii_action', choices=['redact', 'drop'], default='redact')

    args = p.parse_args()

    if args.cmd == 'normalize':
        # kept for compatibility: just normalize each text
        out = []
        for doc in read_jsonl(args.input):
            t = normalize_text(doc.get('text', ''))
            if not t:
                continue
            doc['text'] = t
            doc['hash'] = sha256_of_text(t)
            out.append(doc)
        write_jsonl(args.output, out)
        print('Normalized ->', args.output)

    elif args.cmd == 'dedupe_exact':
        dedupe_exact(args.input, args.output)

    elif args.cmd == 'dedupe_minhash':
        dedupe_minhash(args.input, args.output, threshold=args.threshold, num_perm=args.num_perm)

    elif args.cmd == 'filter_lang':
        filter_lang(args.input, args.output, target_lang=args.lang, prob_threshold=args.prob)

    elif args.cmd == 'filter_pii':
        filter_or_redact_pii(args.input, args.output, action=args.action)

    elif args.cmd == 'split':
        split_dataset(args.input, args.out_dir, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    elif args.cmd == 'train_tokenizer':
        train_sentencepiece(args.input, args.model_prefix, vocab_size=args.vocab_size, model_type=args.model_type)

    elif args.cmd == 'tokenize_shard':
        tokenize_and_shard(args.input, args.sp_model, args.out_dir, samples_per_shard=args.samples_per_shard)

    elif args.cmd == 'download':
        if args.source == 'hf':
            download_hf_dataset(args.dataset, args.output, split=args.split, text_field=args.text_field, limit=args.limit, config=args.config)
        elif args.source == 'wikipedia':
            download_wikipedia(args.query, args.output, limit=args.limit, language=args.lang)
        elif args.source == 'arxiv':
            download_arxiv(args.query, args.output, limit=args.limit)
        elif args.source == 'urls':
            if not args.url_list:
                raise ValueError('For source=urls, pass --url_list pointing to a file with one URL per line')
            download_url_list(args.url_list, args.output, per_page_limit=args.limit)

    elif args.cmd == 'download_all':
        srcs = [s.strip().lower() for s in args.sources.split(',') if s.strip()]
        download_sources(srcs, args.out_dir, limit_per_source=args.limit_per_source, dataset_split=args.split, hf_config=args.dataset, wikipedia_query=args.query, arxiv_query=args.query, url_list=args.url_list, language=args.lang)

    elif args.cmd == 'run_all':
        run_all_pipeline(args.input, args.out_dir, args.tmp_dir, lang=args.lang, samples_per_shard=args.samples_per_shard, pii_action=args.pii_action)

    else:
        p.print_help()


if __name__ == '__main__':
    main()
