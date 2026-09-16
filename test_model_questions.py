#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_questions(model_name, questions, max_new_tokens=80, temperature=0.7, top_p=0.9, seed=42):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    results = []
    for idx, question in enumerate(questions, 1):
        messages = [{"role": "user", "content": question}]
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

        answer = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        item = {
            'index': idx,
            'question': question,
            'answer': answer,
        }
        results.append(item)
        print(f'Q{idx}: {question}')
        print('A:')
        print(answer)
        print('\n---\n')
    return results


def main():
    parser = argparse.ArgumentParser(description='Simple Q&A evaluation harness for a pretrained local model.')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct', help='Model name on Hugging Face')
    parser.add_argument('--questions_file', type=str, default=None, help='Optional path to a JSONL or TXT file of questions')
    parser.add_argument('--max_new_tokens', type=int, default=80)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--output_jsonl', type=str, default='results/questions_answers.jsonl')
    args = parser.parse_args()

    if args.questions_file:
        path = Path(args.questions_file)
        if path.suffix.lower() == '.jsonl':
            with path.open('r', encoding='utf-8') as f:
                questions = [json.loads(line)['question'] for line in f if line.strip()]
        else:
            with path.open('r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]
    else:
        questions = [
            'What is artificial intelligence in one paragraph?',
            'Explain the difference between AI and machine learning.',
            'Write a short Python function to check if a number is prime.',
            'What is the capital of France?',
            'Give me three tips for learning programming quickly.',
        ]

    results = run_questions(
        model_name=args.model_name,
        questions=questions,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    out = Path(args.output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Wrote {len(results)} answers to {out}')


if __name__ == '__main__':
    main()
