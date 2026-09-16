#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_reply(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def run_chat(model_name='Qwen/Qwen2.5-0.5B-Instruct', max_new_tokens=200, temperature=0.7, top_p=0.9, seed=42):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    print('Qwen local chat started. Type "exit" or "quit" to end.')
    messages = []
    while True:
        user_input = input('You: ').strip()
        if user_input.lower() in {'exit', 'quit'}:
            print('Goodbye!')
            break
        if not user_input:
            continue

        messages.append({'role': 'user', 'content': user_input})
        reply = generate_reply(model, tokenizer, user_input, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        print('Assistant:', reply)
        print()
        messages.append({'role': 'assistant', 'content': reply})


def main():
    parser = argparse.ArgumentParser(description='Main local Qwen chat entrypoint.')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name on Hugging Face')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt to answer without entering chat loop')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.prompt is not None:
        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
        model.eval()
        print(generate_reply(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p))
        return

    run_chat(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
