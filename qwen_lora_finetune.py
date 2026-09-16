#!/usr/bin/env python3
"""Fine-tune a small pretrained causal LM with LoRA using locally prepared text data.

This is the realistic way to build something much smarter than the toy model while
remaining feasible on a local workstation. It does NOT create a full frontier model,
but it is a legitimate path to a strong local model.

Example usage:
  python qwen_lora_finetune.py --model_name Qwen/Qwen2.5-0.5B --data_dir data/cleaned/shards --output_dir outputs/qwen_lora --max_examples 2000 --epochs 1

Notes:
- For best results, use a real instruction dataset later (not raw text alone).
- Requires: pip install peft transformers datasets accelerate sentencepiece
- This script assumes a CUDA-capable GPU if you want reasonable speed.
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def load_texts_from_shards(data_dir: str, max_examples: int = 5000):
    texts = []
    for shard in sorted(Path(data_dir).glob('*.jsonl.gz')):
        with gzip.open(shard, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get('text') or obj.get('content')
                if text and isinstance(text, str) and text.strip():
                    texts.append(text.strip())
                if len(texts) >= max_examples:
                    return texts
    if not texts:
        raise ValueError(f'No text data found in {data_dir}. Run the data pipeline first.')
    return texts


def make_dataset(texts):
    samples = []
    for text in texts:
        # Use a simple instruction-like wrapper for causal LM training.
        # This teaches the model to continue a response from a prompt-like template.
        prompt = f"### Instruction:\nWrite a helpful answer.\n\n### Input:\n{text[:300]}\n\n### Response:\n"
        samples.append({'text': prompt})
    ds = Dataset.from_list(samples)
    return ds


def tokenize(batch, tokenizer, max_length=512):
    enc = tokenizer(batch['text'], truncation=True, max_length=max_length, padding='max_length', return_tensors='pt')
    enc['labels'] = enc['input_ids'].clone()
    return enc


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a small pretrained model with LoRA.')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B', help='Base model to fine-tune')
    parser.add_argument('--data_dir', type=str, default='data/cleaned/shards', help='Directory containing gzipped JSONL shards')
    parser.add_argument('--output_dir', type=str, default='outputs/qwen_lora', help='Where to save the fine-tuned model')
    parser.add_argument('--max_examples', type=int, default=2000, help='Maximum examples to use from the local data shards')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    texts = load_texts_from_shards(args.data_dir, max_examples=args.max_examples)
    ds = make_dataset(texts)
    print(f'Loaded {len(ds)} training samples.')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = ds.map(lambda batch: tokenize(batch, tokenizer, max_length=args.max_length), batched=True, batch_size=32)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to('cuda')

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f'Fine-tuned model saved to {args.output_dir}')


if __name__ == '__main__':
    main()
