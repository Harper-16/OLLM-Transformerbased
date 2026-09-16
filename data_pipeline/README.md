Data pipeline for transformer training (collection & cleaning)

This folder contains a single, self-contained pipeline script and a requirements file to run the full data preparation flow for training language models.

Install
  python -m pip install -r data_pipeline/requirements.txt

Main script: data_pipeline/pipeline.py

Subcommands (run with `python pipeline.py <cmd> --help` for details):
  download            - Download raw data from popular sources: Hugging Face, Wikipedia, arXiv, or a text file of URLs
  download_all        - Download from multiple sources and merge them into a single raw JSONL file
  normalize           - Normalize raw JSONL input into normalized JSONL (strip HTML, unicode NFKC, collapse whitespace)
  dedupe_exact        - Exact deduplication by SHA256 of normalized text
  dedupe_minhash      - Near-duplicate removal using MinHashLSH
  filter_lang         - Keep only documents in the target language (default: en)
  filter_pii          - PII detection and redaction or drop (emails, phones, SSNs)
  split               - Create train/validation/test splits and save indexes with provenance
  train_tokenizer     - Train SentencePiece tokenizer on a text corpus
  tokenize_shard      - Tokenize with SentencePiece and shard into JSONL.gz shards
  run_all             - Run a recommended sequence of all steps in order

Input format
  The pipeline expects newline-delimited JSON (JSONL) where each line is an object with at least the field:
    {"text": "...raw document text...", "source": "source-name", "license": "..."}

Outputs
  The script writes JSONL files with added metadata (hash, language, tags) and sharded tokenized outputs.

Notes
  - The pipeline aims to be practical and readable; for very large corpora you should run each stage in a distributed setting or on a machine with enough RAM/disk.
  - MinHash parameters and thresholds are configurable; tune them for your dataset.

Examples
  # Download from Hugging Face dataset (example: C4)
  python data_pipeline/pipeline.py download --source hf --dataset c4 --split train --limit 100 --output data/raw_hf.jsonl

  # Download Wikipedia pages for a search query
  python data_pipeline/pipeline.py download --source wikipedia --query "artificial intelligence" --limit 20 --output data/raw_wikipedia.jsonl

  # Download arXiv research abstracts for a topic
  python data_pipeline/pipeline.py download --source arxiv --query "transformers" --limit 20 --output data/raw_arxiv.jsonl

  # Download from a text file containing one URL per line
  python data_pipeline/pipeline.py download --source urls --url_list data/urls.txt --limit 50 --output data/raw_urls.jsonl

  # Download from multiple sources at once and combine them into a single raw JSONL file
  python data_pipeline/pipeline.py download_all --out_dir data/downloaded --sources hf,wikipedia,arxiv --dataset c4 --query "large language model" --limit_per_source 50

  # Normalize from a raw JSONL file
  python data_pipeline/pipeline.py normalize --input data/raw.jsonl --output data/normalized.jsonl

  # Exact dedupe
  python data_pipeline/pipeline.py dedupe_exact --input data/normalized.jsonl --output data/dedup_exact.jsonl

  # Run full recommended pipeline
  python data_pipeline/pipeline.py run_all --input data/raw.jsonl --out_dir data/cleaned --tmp_dir data/tmp --lang en --samples_per_shard 5000

Notes
  - download uses public endpoints and may rate-limit or change format over time.
  - For very large downloads, prefer storing the raw files in cloud buckets and streaming them into the cleaning pipeline.

If you want, provide a small sample of your data and the pipeline can be adapted/tuned to match it.
