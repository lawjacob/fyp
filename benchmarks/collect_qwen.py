"""Prepare GSM8K and collect resumable, independently seeded Qwen outcomes."""
import argparse
import gc
import hashlib
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path


def numeric_answer(text, reference=False):
    # Deliberately strict: do not mistake an intermediate number for an answer.
    matches = re.findall(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(',', ''))
    except InvalidOperation:
        return None


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16) % (2**31)


def prepare(args):
    from datasets import load_dataset
    import numpy as np
    ds = load_dataset('openai/gsm8k', 'main')
    rng = np.random.default_rng(args.seed)
    cutoff = float(np.median([len(r['question'].split()) for r in ds['train']]))
    rows = []
    for source, group, count, long in [('train', 'train', args.train, False),
                                      ('test', 'id', args.eval, False),
                                      ('test', 'length_shift', args.eval, True)]:
        pool = [i for i, r in enumerate(ds[source]) if (len(r['question'].split()) > cutoff) == long]
        if count > len(pool):
            raise ValueError(f'{group}: requested {count}, only {len(pool)} available')
        for i in rng.choice(pool, count, replace=False):
            r = ds[source][int(i)]
            rows.append(dict(id=f'gsm8k/{source}/{i}', prompt=r['question'], answer=r['answer'], split=group))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(r) + '\n' for r in rows)
    if out.exists() and out.read_text() != content:
        raise ValueError('Existing prompt file differs; use a new output path')
    out.write_text(content)
    print(f'Prepared {len(rows)} prompts; length cutoff={cutoff}. Length shift is NOT a new semantic domain.', flush=True)


def collect(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available() and args.device != 'cpu':
        raise RuntimeError('CUDA unavailable. Check PyTorch/CUDA installation or use --device cpu for a tiny smoke run.')
    rows = read_jsonl(args.prompts)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prompt IDs')
    if any(numeric_answer(r['answer']) is None for r in rows):
        raise ValueError('Every reference must have a numeric #### answer')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(models=args.models, samples=args.samples, seed=args.seed,
                  max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
                  prompt_sha256=hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
                  collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  torch=torch.__version__, transformers=transformers.__version__, device=args.device,
                  instruction='Solve the problem. Finish with exactly #### followed by the numeric answer.')
    manifest = out / 'collection.json'
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError('Collection configuration changed; use a different output directory')
    manifest.write_text(json.dumps(config, indent=2))
    # Separate model files permit resuming without regenerating completed records.
    for model_index, model_name in enumerate(args.models):
        path = out / f'model_{model_index}.jsonl'
        done = set()
        if path.exists():
            # A killed process may leave only its final JSONL line incomplete.
            with path.open('rb+') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        if f.read():
                            raise ValueError(f'Corruption before final line in {path}')
                        f.truncate(offset)
                        break
                    done.add((record['id'], record['sample']))
                    if not line.endswith(b'\n'):
                        f.write(b'\n')
        pending = [(r, s) for r in rows for s in range(args.samples) if (r['id'], s) not in done]
        if not pending:
            print(f'{model_name}: already complete', flush=True)
            continue
        print(f'Loading {model_name}; {len(pending)} generations remaining', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float32 if args.device == 'cpu' else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(args.device).eval()
        print(f'Resolved model revision: {getattr(model.config, "_commit_hash", None)}', flush=True)
        with path.open('a', buffering=1) as f:
            for index, (row, sample) in enumerate(pending):
                seed = stable_seed(args.seed, model_name, row['id'], sample)
                set_seed(seed)
                messages = [{'role': 'system', 'content': config['instruction']},
                            {'role': 'user', 'content': row['prompt']}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors='pt').to(args.device)
                start = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=True, temperature=args.temperature,
                                               top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                                               pad_token_id=tokenizer.eos_token_id)
                tokens = generated[0, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(tokens, skip_special_tokens=True)
                answer = numeric_answer(response)
                record = dict(id=row['id'], model=model_name, sample=sample, seed=seed,
                              success=int(answer is not None and answer == numeric_answer(row['answer'])),
                              parse_ok=answer is not None, response=response, output_tokens=len(tokens),
                              input_tokens=inputs['input_ids'].shape[1], latency_seconds=time.monotonic()-start,
                              truncated=len(tokens) == args.max_new_tokens and int(tokens[-1]) != tokenizer.eos_token_id,
                              model_revision=getattr(model.config, '_commit_hash', None))
                f.write(json.dumps(record) + '\n')
                if (index + 1) % 10 == 0 or index == 0:
                    print(f'{model_name}: {index+1}/{len(pending)} remaining jobs completed', flush=True)
        del model, tokenizer, inputs, generated, tokens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def pack(args):
    import numpy as np
    from sklearn.feature_extraction.text import HashingVectorizer
    rows = read_jsonl(args.prompts)
    source = Path(args.collection)
    manifest = json.loads((source / 'collection.json').read_text())
    if hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest() != manifest['prompt_sha256']:
        raise ValueError('Prompt file differs from collection manifest')
    models, samples = manifest['models'], manifest['samples']
    if len(args.costs) != len(models) or min(args.costs) <= 0:
        raise ValueError('Supply one positive fixed cost per model')
    ids = {r['id']: i for i, r in enumerate(rows)}
    outcomes = np.full((len(models), len(rows), samples), -1, dtype=np.int8)
    for m, name in enumerate(models):
        records = read_jsonl(source / f'model_{m}.jsonl')
        for r in records:
            i, s = ids[r['id']], r['sample']
            if r['model'] != name or not 0 <= s < samples or outcomes[m, i, s] != -1:
                raise ValueError('Invalid or duplicate outcome record')
            if r['success'] not in (0, 1):
                raise ValueError('Success labels must be binary')
            outcomes[m, i, s] = r['success']
        print(f'{name}: pass@1={outcomes[m].mean():.3f}, '
              f'parse failures={sum(not r["parse_ok"] for r in records)}, '
              f'truncated={sum(r["truncated"] for r in records)}', flush=True)
    if np.any(outcomes < 0):
        raise ValueError('Incomplete collection; rerun collect first')
    # Stateless features: no fit on evaluation prompts. An explicit intercept is added.
    features = HashingVectorizer(n_features=args.dim-1, alternate_sign=False, norm='l2').transform(
        [r['prompt'] for r in rows]).toarray()
    features = np.column_stack([features, np.ones(len(rows))])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, contexts=features, outcomes=outcomes, costs=args.costs,
                        ids=[r['id'] for r in rows], splits=[r['split'] for r in rows], models=models,
                        metadata=json.dumps(dict(collection=manifest, feature='hashing+intercept',
                                                 cost_note=args.cost_note)))
    print(f'Packed {outcomes.shape} into {out}; costs are {args.cost_note}', flush=True)


def main():
    from benchmarks.cache_locks import configure_cache_locks
    configure_cache_locks()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--output', required=True)
    a.add_argument('--train', type=int, default=300)
    a.add_argument('--eval', type=int, default=100)
    a.add_argument('--seed', type=int, default=42)
    a = sub.add_parser('collect')
    a.add_argument('--prompts', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--models', nargs='+', default=['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct'])
    a.add_argument('--samples', type=int, default=5)
    a.add_argument('--max-new-tokens', type=int, default=512)
    a.add_argument('--temperature', type=float, default=0.7)
    a.add_argument('--top-p', type=float, default=0.9)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--device', default='cuda:0')
    a = sub.add_parser('pack')
    a.add_argument('--prompts', required=True)
    a.add_argument('--collection', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--costs', nargs='+', type=float, required=True)
    a.add_argument('--cost-note', default='user-supplied relative per-call cost proxy, not dollars')
    a.add_argument('--dim', type=int, default=32)
    args = p.parse_args()
    if getattr(args, 'samples', 1) < 1 or getattr(args, 'dim', 2) < 2:
        p.error('samples must be positive and dim >= 2')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
