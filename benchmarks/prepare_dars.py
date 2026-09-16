"""Prepare an audited, fixed-prompt DARS retry table; no model inference.

Uses only source-training rewrite 0 (five decodes at temperature 0.7).
Published test decoding rows use different temperatures and are excluded.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

REVISION = '3109e08fb15b24192c0284b9fc11ca9f71110950'
MODELS = [
    'google/gemma-3-12b-it',
    'mistralai/mistral-small-3.2-24b-instruct',
    'qwen/qwen3-32b',
    'meta-llama/llama-3.3-70b-instruct',
    'google/gemini-2.5-flash-lite',
    'deepseek/deepseek-chat-v3.1',
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(source, output, dataset='gpqa', train_count=140, seed=42, dim=32):
    output = Path(output)
    if (output / 'outcomes.npz').exists():
        raise ValueError('Output already exists; use a new output directory')
    groups = defaultdict(list)
    generation_ids, response_ids = set(), set()
    all_rows = 0
    with Path(source).open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            all_rows += 1
            if r['dataset'] != dataset or r['split'] != 'train':
                raise ValueError('Expected the selected dataset source-training file only')
            if r.get('prompt_variant_source') != 'rewrite' or r.get('prompt_variant_id') != 0:
                continue
            if r['model'] not in MODELS:
                raise ValueError('Unexpected model; review the dataset revision')
            if r['score'] not in (0, 1):
                raise ValueError('Only published binary correctness scores are supported')
            expected_type = 'binary_mcq_accuracy' if dataset == 'gpqa' else 'binary_math_exact_match'
            if r['score_type'] != expected_type:
                raise ValueError(f'Unexpected score type: {r["score_type"]}')
            if r['request_config'] != dict(temperature=.7, top_p=.95,
                                            max_tokens=768 if dataset == 'gpqa' else 1536):
                raise ValueError('Mixed or unexpected decoding settings')
            for key, seen in [('generation_key', generation_ids), ('openrouter_response_id', response_ids)]:
                value = r.get(key)
                if not value or value in seen:
                    raise ValueError(f'Missing or duplicate {key}')
                seen.add(value)
            cost = r.get('usage', {}).get('cost')
            if cost is None or not np.isfinite(cost) or cost <= 0:
                raise ValueError('Missing/nonpositive raw usage.cost')
            groups[r['query_id']].append(r)
    ids = sorted(groups)
    if not 0 < train_count < len(ids) or dim < 2:
        raise ValueError('Need nonempty train/evaluation splits and dim >= 2')
    outcomes = np.empty((len(MODELS), len(ids), 5), dtype=np.int8)
    raw_cost = np.empty_like(outcomes, dtype=float)
    texts, parent_texts, categories, selected = [], set(), [], []
    for i, qid in enumerate(ids):
        rows = groups[qid]
        if len(rows) != len(MODELS) * 5:
            raise ValueError(f'Incomplete question/model coverage: {qid}')
        # Includes answer choices, but never the correct answer or response text.
        prompt_signatures = {json.dumps([r['input_question'], r.get('context'), r.get('choices')]) for r in rows}
        originals = {' '.join(r['original_question'].split()).casefold() for r in rows}
        if len(prompt_signatures) != 1 or len(originals) != 1:
            raise ValueError(f'Inconsistent prompts across models/decodes: {qid}')
        original = next(iter(originals))
        if original in parent_texts:
            raise ValueError('Duplicate parent question under different IDs')
        parent_texts.add(original)
        texts.append(next(iter(prompt_signatures)))
        categories.append(str(rows[0].get('category', 'unknown')))
        for g, model in enumerate(MODELS):
            samples = sorted((r for r in rows if r['model'] == model), key=lambda r: r['decode_id'])
            if [r['decode_id'] for r in samples] != list(range(5)):
                raise ValueError(f'Missing or duplicate decodes: {qid}, {model}')
            outcomes[g, i] = [r['score'] for r in samples]
            raw_cost[g, i] = [r['usage']['cost'] for r in samples]
        selected.extend(rows)
    order = np.random.default_rng(seed).permutation(len(ids))
    splits = np.full(len(ids), 'id', dtype='<U5')
    splits[order[:train_count]] = 'train'
    train = splits == 'train'
    mean_cost = raw_cost[:, train].mean(axis=(1, 2))
    unit = float(mean_cost.min())
    costs = mean_cost / unit
    x = HashingVectorizer(n_features=dim-1, alternate_sign=False, norm='l2').transform(texts).toarray()
    x = np.column_stack([x, np.ones(len(ids))])
    audit = dict(dataset=dataset, revision=REVISION, source_sha256=digest(source),
                 importer_sha256=digest(__file__), source_rows=all_rows, selected_rows=len(selected),
                 selection='source train, rewrite 0, decode IDs 0..4; new parent-level split',
                 published_test_excluded='Its decoding_variation observations mix temperatures.',
                 split_seed=seed, unique_parents=len(ids), train_parents=int(train.sum()),
                 evaluation_parents=int((~train).sum()), models=MODELS, samples_per_model_prompt=5,
                 cost_definition='Fixed mean raw usage.cost on NEW training parents, divided by cheapest mean. '
                                 'Expected-cost proxy; not actual spend or latency per replay attempt.',
                 cost_unit_recorded_dollars=unit, train_mean_recorded_dollars=mean_cost.tolist(),
                 fixed_costs=costs.tolist(), features=f'{dim-1} hashed features + intercept',
                 categories_train=dict(Counter(np.array(categories)[train])),
                 categories_evaluation=dict(Counter(np.array(categories)[~train])),
                 decoding=selected[0]['request_config'],
                 score='Published binary score, not independently rejudged; no GSM8K #### parsing',
                 model_audit={})
    for g, model in enumerate(MODELS):
        rows = [r for r in selected if r['model'] == model]
        audit['model_audit'][model] = dict(records=len(rows),
            train_success=float(outcomes[g, train].mean()), eval_success=float(outcomes[g, ~train].mean()),
            truncated=sum(r.get('finish_reason') == 'length' for r in rows),
            missing_prediction=sum(r.get('predicted_answer') is None for r in rows),
            empty_output=sum(not r.get('output_text') for r in rows),
            parse_flag_false=sum(r.get('parse_success') is False for r in rows))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    with (output / 'prompts.jsonl').open('w') as f:
        for i, qid in enumerate(ids):
            f.write(json.dumps(dict(id=qid, split=str(splits[i]), prompt=texts[i], category=categories[i]))+'\n')
    np.savez_compressed(output / 'outcomes.npz', contexts=x, outcomes=outcomes, costs=costs,
                        ids=ids, splits=splits, models=MODELS, metadata=json.dumps(audit))
    print(json.dumps(audit, indent=2))
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['gpqa', 'math-500'], default='gpqa')
    p.add_argument('--output', required=True)
    p.add_argument('--source-file', help='Use an already downloaded source-training JSONL')
    p.add_argument('--train-count', type=int, default=140)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--dim', type=int, default=32)
    args = p.parse_args()
    if (Path(args.output) / 'outcomes.npz').exists():
        p.error('Output already exists; use a new directory')
    if args.source_file:
        source = Path(args.source_file)
    else:
        raw = Path(args.output) / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        source = raw / f'{args.dataset}-{REVISION}-train.jsonl'
        if not source.exists():
            url = f'https://huggingface.co/datasets/AIGNLAI/DARS/resolve/{REVISION}/{args.dataset}/train_scored_generations.jsonl'
            print(f'Downloading public {args.dataset} responses; no inference.', flush=True)
            temp = source.with_suffix('.part')
            with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as f:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    f.write(block)
            temp.replace(source)
    prepare(source, args.output, args.dataset, args.train_count, args.split_seed, args.dim)


if __name__ == '__main__':
    main()
