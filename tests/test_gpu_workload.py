from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from inference_lab import benchmark
from inference_lab.metrics import RequestResult

ROOT = Path(__file__).resolve().parents[1]


class GPUWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'configs/gpu-experiment.json').read_text())
        self.rows, _ = benchmark.load_workload(ROOT / self.config['workload'])

    def test_fixed_campaign_configuration(self):
        benchmark.validate_config(self.config)
        self.assertEqual(self.config['backend'], 'vllm')
        self.assertEqual(self.config['seed'], 42)
        self.assertEqual(self.config['num_requests'], 120)
        self.assertEqual(self.config['max_tokens'], 192)
        self.assertEqual(self.config['warmup_requests'], 8)
        self.assertFalse(self.config['ignore_eos'])
        cases = 3 * 3 * 3
        self.assertEqual(cases * self.config['num_requests'], 3240)
        self.assertEqual(cases * self.config['warmup_requests'], 216)

    def test_corpus_has_unique_prompts_across_distinct_input_lengths(self):
        self.assertEqual(len(self.rows), 24)
        self.assertEqual(len({row['id'] for row in self.rows}), 24)
        self.assertEqual(len({row['prompt'] for row in self.rows}), 24)
        self.assertEqual(Counter(row['input_band'] for row in self.rows),
                         {'short': 8, 'medium': 8, 'long': 8})
        self.assertEqual({row['input_band'] for row in self.rows[:self.config['warmup_requests']]},
                         {'short', 'medium', 'long'})
        bounds = {'short': (1, 400), 'medium': (401, 850), 'long': (851, 1500)}
        for row in self.rows:
            with self.subTest(case=row['id']):
                low, high = bounds[row['input_band']]
                self.assertTrue(low <= len(row['prompt'].encode('utf-8')) <= high)

    def test_generated_requests_fit_conservative_context_budget(self):
        sent = []

        def complete(base_url, model, prompt, max_tokens, timeout, request_id, ignore_eos):
            sent.append((request_id, prompt, max_tokens))
            return RequestResult(request_id=request_id, success=True, status_code=200,
                                 prompt_tokens=1, output_tokens=1, ttft_s=0.01,
                                 e2e_s=0.02, text='fixture')

        config = {**self.config, 'workload': str(ROOT / self.config['workload'])}
        with patch.object(benchmark, 'check_backend'), \
                patch.object(benchmark, 'git_revision', return_value=None), \
                patch.object(benchmark, 'request_completion', side_effect=complete):
            result = benchmark.run_benchmark(config)
        self.assertEqual(result['summary']['attempted'], 120)
        self.assertEqual(len(sent), 128)
        for request_id, prompt, output_limit in sent:
            with self.subTest(request_id=request_id):
                # Byte-based headroom is an offline guard; check actual tokenizer counts on the GPU host.
                self.assertLessEqual(len(prompt.encode('utf-8')) + 128 + output_limit, 2048)
        measured = [prompt for request_id, prompt, _ in sent if request_id >= 0]
        bands = {row['prompt']: row['input_band'] for row in self.rows}
        observed = Counter(bands[prompt.rsplit('\nRequest tag: ', 1)[0]] for prompt in measured)
        self.assertEqual(set(observed), {'short', 'medium', 'long'})
        self.assertTrue(all(count >= 25 for count in observed.values()))


if __name__ == '__main__':
    unittest.main()
