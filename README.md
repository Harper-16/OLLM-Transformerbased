# OLLM-Transformerbased (AB-model)

This repository holds a collection of NumPy-based transformer prototypes, data ingestion tools, and memory/retrieval components developed for low-resource devices (Raspberry Pi prototypes) and experimentation. The project is intentionally structured to keep source, scripts, models, and documentation separate while preserving all original files.

## Repository layout

- src/  : Primary Python source files (transformer prototypes, training & generation scripts)
- scripts/ : Utility shell scripts and helper run scripts
- data/ : Data, corpora, temporary downloads, and cleaned training corpora
- models/ : Large model weight files, SQLite memory DBs, and other heavy artifacts
- docs/ : Documentation and notes (empty placeholder)

Files were moved into these folders to make the repository easier to navigate. No files were removed.

## Key files and descriptions

- src/improved_numpy_ai.py
  - Practical NumPy-based model, training loop with Adam, memory DB integration, and generation utilities.

- src/proper_numpy_transformer_memory.py
  - More “proper” transformer prototype with local memory/retrieval (SQLite) and improved architecture.

- src/pi_zero_tiny_transformer.py, src/pi_zero_word_transformer.py
  - Tiny, Pi-oriented prototypes (character- and word-level) designed for very low-resource devices.

- src/pi_numpy_smollm.py
  - Attempted NumPy forward-pass for a SmolLM-style model after weight conversion.

- src/model.py
  - Shared model components / utility classes used across prototypes.

- src/internet_data_loop.py, src/build_training_corpus.py
  - Utilities to fetch and clean web text and build local training corpora. Be careful when running crawlers — respect robots.txt and copyright.

- scripts/pi_real_assistant.sh
  - Helper to build / run llama.cpp on supported hardware (Pi 4/5 or x86), kept for reference.

- models/np, models/re, models/.memory.sqlite, models/.improved_memdb.json
  - Large artifacts and DBs moved to models/ to keep the repo root tidy.

## How to run (examples)

- Train or run the improved NumPy AI prototype:

  python3 src/improved_numpy_ai.py --prompt "hello" --epochs 1

- Run a tiny Pi transformer (example):

  python3 src/pi_zero_tiny_transformer.py --data data/ --steps 80 --prompt "hello"

- Build a cleaned training corpus (example):

  python3 src/build_training_corpus.py --input-dir data/raw --output data/training_corpus.txt

## Notes and recommendations

- Large binary files and DBs were moved to `models/`. If you plan to push this repository to a remote, consider adding any very large artifacts to `.gitignore` and using Git LFS instead.

- This repository contains experimental code and prototype scripts. Some scripts assume being run from the repo root — invoking them with `python3 src/<script>.py` will work after reorganization.

- If you want me to update internal import paths in the Python files (e.g., to change relative imports), I can do that next. Currently scripts are left unmodified and should run by pointing the Python command at the files in `src/`.

- Keep backups of any trained DBs before running tests that rewrite them.

## Contributing

If you want additional reorganization (e.g., adding a package layout, installing with `pip install -e .`, or creating a single entrypoint `cli.py`), say the word and I will make the changes.

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
