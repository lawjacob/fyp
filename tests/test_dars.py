import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.prepare_dars import MODELS, prepare
from benchmarks.summarize_paired import summarize


class DarsTests(unittest.TestCase):
    def fixture(self):
        return [dict(dataset='gpqa', split='train', query_id=f'q{i}', model=model,
                     prompt_variant_source='rewrite', prompt_variant_id=0, decode_id=d,
                     original_question=f'Original question {i}', input_question=f'Question {i}',
                     choices=['yes', 'no'], context=None, category='Physics',
                     score_type='binary_mcq_accuracy', score=int(d % 2 == 0),
                     request_config=dict(temperature=.7, top_p=.95, max_tokens=768),
                     generation_key=f'{i}-{g}-{d}', openrouter_response_id=f'{i}-{g}-{d}',
                     usage={'cost': float(g+1)*(1+i)}, output_text='yes', predicted_answer='A',
                     finish_reason='stop', parse_success=True)
                for i in range(4) for g, model in enumerate(MODELS) for d in range(5)]

    def run_prepare(self, rows, root):
        source = Path(root)/'source.jsonl'
        source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return prepare(source, Path(root)/'out', train_count=2)

    def test_complete_table_parent_split_train_only_cost(self):
        with tempfile.TemporaryDirectory() as root:
            audit = self.run_prepare(self.fixture(), root)
            with np.load(Path(root)/'out/outcomes.npz') as z:
                self.assertEqual(z['outcomes'].shape, (6, 4, 5))
                self.assertEqual(sum(z['splits'] == 'train'), 2)
                np.testing.assert_allclose(z['costs'], np.arange(1, 7))
                train_ids = [int(q[1:]) for q in z['ids'][z['splits'] == 'train']]
                self.assertAlmostEqual(audit['cost_unit_recorded_dollars'], np.mean(np.array(train_ids)+1))

    def test_reject_missing_duplicate_mixed_settings_and_changed_prompt(self):
        for kind in ('missing', 'duplicate', 'settings', 'prompt'):
            rows = self.fixture()
            if kind == 'missing':
                rows.pop()
            elif kind == 'duplicate':
                rows.append(rows[0])
            elif kind == 'settings':
                rows[0]['request_config']['temperature'] = .2
            else:
                rows[0]['input_question'] = 'Different question'
            with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
                self.run_prepare(rows, root)

    def test_intervals_pair_by_question_across_seeds(self):
        with tempfile.TemporaryDirectory() as root:
            for seed in (11, 22):
                with (Path(root)/f'seed_{seed}.csv').open('w') as f:
                    w = csv.DictWriter(f, fieldnames=['history','scenario','mode','attempt_budget',
                        'prompt_id','seed','policy','success','cost'])
                    w.writeheader()
                    for q in ('a', 'b'):
                        for policy in ('original','ewma','repeat_initial','switch_on_failure'):
                            w.writerow(dict(history=50,scenario='id',mode='episodic',attempt_budget=3,
                                prompt_id=q,seed=seed,policy=policy,success=int(policy=='ewma'),cost=1))
            summarize(root, 50)
            with (Path(root)/'paired_intervals.csv').open() as f:
                rows = list(csv.DictReader(f))
            ewma = next(r for r in rows if r['policy'] == 'ewma')
            self.assertEqual(ewma['unique_questions'], '2')
            self.assertEqual(float(ewma['success_delta_vs_original_low']), 1.)


if __name__ == '__main__':
    unittest.main()
