#!/usr/bin/env python3
"""Lightweight Pi-friendly chat script using a small Hugging Face model.

This is the best pure-Python local option for a Raspberry Pi.
It avoids llama.cpp and keeps the model in standard Transformers.

Recommended defaults:
- Pi Zero W: HuggingFaceTB/SmolLM2-135M-Instruct
- Pi 4/5: HuggingFaceTB/SmolLM2-360M-Instruct

Run examples:
  python pi_chat.py
  python pi_chat.py --model_name HuggingFaceTB/SmolLM2-360M-Instruct --max_new_tokens 40
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_model(model_name: str):
    # Keep CPU work predictable and lighter on Pi hardware.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to('cpu')
    model.eval()
    return tokenizer, model


def generate_reply(model, tokenizer, user_text: str, max_new_tokens: int = 32, temperature: float = 0.8, top_p: float = 0.9):
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def chat_loop(model_name: str, max_new_tokens: int, temperature: float, top_p: float, seed: int):
    torch.manual_seed(seed)
    tokenizer, model = build_model(model_name)

    print(f'Loaded model: {model_name}')
    print('Chat started. Type "exit" or "quit" to end.')
    messages = []

    while True:
        user_text = input('You: ').strip()
        if user_text.lower() in {'exit', 'quit'}:
            print('Goodbye!')
            break
        if not user_text:
            continue

        messages.append({'role': 'user', 'content': user_text})
        reply = generate_reply(model, tokenizer, user_text, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        print('Assistant:', reply)
        print()
        messages.append({'role': 'assistant', 'content': reply})


def main():
    parser = argparse.ArgumentParser(description='Pi-friendly local chat with a small Hugging Face model.')
    parser.add_argument('--model_name', type=str, default='HuggingFaceTB/SmolLM2-135M-Instruct', help='Small model to run locally on Pi')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='Short answers are faster and better for Pi')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    chat_loop(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
