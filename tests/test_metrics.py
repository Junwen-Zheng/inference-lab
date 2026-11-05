import math
import unittest

from inference_lab.metrics import RequestResult, distribution_ms, percentile, summarize


class MetricsTests(unittest.TestCase):
    def test_percentiles_interpolate_and_include_endpoints(self):
        self.assertAlmostEqual(percentile([1, 2, 10], 95), 9.2)
        self.assertEqual(percentile([10, 1, 2], 0), 1)
        self.assertEqual(percentile([10, 1, 2], 100), 10)
        self.assertEqual(percentile([7], 99), 7)

    def test_empty_and_missing_latency_samples_stay_unavailable(self):
        self.assertIsNone(percentile([], 99))
        self.assertEqual(distribution_ms([None]),
                         {"mean": None, "p50": None, "p95": None, "p99": None})
        self.assertEqual(distribution_ms([None, .1])["p95"], 100)

    def test_invalid_percentiles_and_latency_samples_are_rejected(self):
        for q in (-1, 101, math.nan, math.inf):
            with self.subTest(q=q), self.assertRaises(ValueError):
                percentile([1], q)
        for value in (-1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                distribution_ms([value])

    def test_failures_do_not_enter_successful_throughput_or_latency(self):
        rows = [RequestResult(0, success=True, output_tokens=10, ttft_s=.1, e2e_s=.5),
                RequestResult(1, success=True, output_tokens=20, ttft_s=.3, e2e_s=1),
                RequestResult(2, success=False, output_tokens=999, ttft_s=10, e2e_s=10)]
        summary = summarize(rows, 2, 200, 1000)
        self.assertEqual(summary["attempted"], 3)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["requests_per_second"], 1)
        self.assertEqual(summary["output_tokens_per_second"], 15)
        self.assertEqual(summary["slo"]["goodput_requests_per_second"], .5)
        self.assertEqual(summary["ttft_ms"]["p50"], 200)
        self.assertAlmostEqual(summary["error_rate"], 1 / 3)

    def test_all_failure_run_has_no_successful_latency(self):
        summary = summarize([RequestResult(0)], 1, 100, 1000)
        self.assertEqual(summary["error_rate"], 1)
        self.assertEqual(summary["output_tokens_per_second"], 0)
        self.assertIsNone(summary["ttft_ms"]["p95"])
        self.assertEqual(summary["slo"]["good_requests"], 0)

    def test_empty_run_has_zero_counts_and_missing_latency(self):
        summary = summarize([], 1, 100, 1000)
        self.assertEqual(summary["attempted"], 0)
        self.assertEqual(summary["requests_per_second"], 0)
        self.assertEqual(summary["error_rate"], 0)
        self.assertIsNone(summary["e2e_ms"]["mean"])

    def test_slo_boundaries_are_inclusive_and_missing_ttft_cannot_pass(self):
        rows = [RequestResult(0, success=True, output_tokens=1, ttft_s=.1, e2e_s=1),
                RequestResult(1, success=True, output_tokens=1, ttft_s=None, e2e_s=.5),
                RequestResult(2, success=True, output_tokens=1, ttft_s=.101, e2e_s=.5)]
        summary = summarize(rows, 2, 100, 1000)
        self.assertEqual(summary["slo"]["good_requests"], 1)
        self.assertIsNone(summary["tpot_ms"]["p95"])

    def test_duration_and_slo_limits_must_be_finite(self):
        for duration in (0, -1, math.nan, math.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                summarize([], duration, 100, 1000)
        for limit in (-1, math.nan, math.inf):
            for limits in ((limit, 1000), (100, limit)):
                with self.subTest(limits=limits), self.assertRaises(ValueError):
                    summarize([], 1, *limits)

    def test_unknown_successful_token_count_is_not_fabricated(self):
        result = summarize([RequestResult(0, success=True, output_tokens=None)],
                           1, 100, 1000)
        self.assertIsNone(result["output_tokens_per_second"])

    def test_invalid_token_counts_are_rejected(self):
        for tokens in (-1, 1.5, True):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                summarize([RequestResult(0, success=True, output_tokens=tokens)],
                          1, 100, 1000)


if __name__ == "__main__":
    unittest.main()
