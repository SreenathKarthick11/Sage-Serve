"""Regression tests for the failure cases documented in the inference review.

Run: PYTHONPATH=mlops python -B -m unittest discover -s mlops/inference/tests -v
"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from inference.forecast_evaluation import evaluate_forecasts
from inference.loader import DETAIL_COLUMNS, InputError, load_results
from inference.metrics import _minmax, build_report, normalize_weights
from inference.run_inference import main


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"archived_runs"
        self.root.mkdir()

    def experiment(self, *, asynchronous="True", model="A", region="westus", run=0):
        leaf = (self.root/str(run)/"demo"/f"feed_async_{asynchronous}"/"feed_async_granularity_1"/
                "scaling_level_2"/"scaling_interval_-1"/"lts_False_stsTrue"/"arbiter_basic_arbiter")
        self.details(leaf, 0, [4., 6., 8.], model=model, region=region)
        return leaf

    def details(self, leaf, window, responses, *, model="A", region="westus"):
        app = f"{region}_{model}_0"
        folder = leaf/"detailed"/app
        folder.mkdir(parents=True, exist_ok=True)
        responses = np.asarray(responses, dtype=float)
        ttft = np.minimum(1., responses/2)
        frame = pd.DataFrame({
            "response_times": responses, "queue_times": np.full(len(responses), .1),
            "ttft_times": ttft, "tbt_times": (responses-ttft)/10,
            "prompt_sizes": np.arange(1, len(responses)+1)*10, "token_sizes": np.full(len(responses), 10),
            "nth_token_overheads": np.full(len(responses), -.02),
        })
        frame.to_csv(folder/f"{window}.csv", index=False)
        (leaf/"summary").mkdir(exist_ok=True)
        pd.DataFrame({"application_id": [app], "response_times_mean": [responses.mean() if len(responses) else -1]}).to_csv(
            leaf/"summary"/f"{window}.csv", index=False)

    def cost(self, leaf, window, times, *, prod=2., model="A"):
        folder = leaf/"arbiters"/"westus"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": times, "model": [model]*len(times),
                      "prod": [prod]*len(times), "spot": [0]*len(times),
                      "memory_util": [.5]*len(times)}).to_csv(folder/f"{window}.csv", index=False)

    def boundaries(self, mapping):
        exps = {str(e.path): e.meta["experiment_id"] for e in load_results(self.root).experiments}
        rows = [{"experiment_id": exps[str(leaf)], "window": w, "start_seconds": start, "end_seconds": end}
                for leaf, bounds in mapping.items() for w, start, end in bounds]
        path = Path(self.temp.name)/"bounds.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def report(self, **kwargs):
        return build_report(self.root, **kwargs)

    def test_clean_percentiles_and_precise_metric_names(self):
        self.experiment()
        row = self.report().metrics.iloc[0]
        self.assertEqual(row.n_requests, 3)
        self.assertAlmostEqual(row.e2e_p50, 6)
        self.assertAlmostEqual(row.e2e_p95, 7.8)
        self.assertAlmostEqual(row.e2e_p99, 7.96)
        self.assertAlmostEqual(row.avg_tbt_p99, .696)
        self.assertEqual(row.tbt_p99, row.avg_tbt_p99)
        self.assertTrue(row.request_data_complete)

    def test_configurations_remain_separate(self):
        first = self.experiment()
        second = self.experiment(asynchronous="False")
        self.details(second, 0, [40, 60, 80])
        rows = self.report().metrics
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows.experiment_id.nunique(), 2)
        self.assertEqual(set(rows.e2e_p50), {6, 60})
        self.assertEqual(set(rows.feed_async), {"True", "False"})
        self.assertEqual(set(rows.feed_async_granularity), {"1"})
        self.assertIn(str(first), set(rows.source_path))

    def test_full_configuration_is_hashed_and_retained(self):
        leaf = self.experiment()
        (leaf/".hydra").mkdir()
        cfg = leaf/".hydra"/"config.yaml"
        cfg.write_text("seed: 0\nextra_setting: first\n")
        one = self.report()
        cfg.write_text("seed: 0\nextra_setting: second\n")
        two = self.report()
        self.assertNotEqual(one.metrics.experiment_id.iloc[0], two.metrics.experiment_id.iloc[0])
        self.assertEqual(two.metadata["experiments"][0]["configuration"]["extra_setting"], "second")

    def test_model_underscores_are_preserved(self):
        self.experiment(model="llama_3_8b")
        self.assertEqual(self.report().metrics.model.iloc[0], "llama_3_8b")

    def test_region_underscores_use_saved_configuration(self):
        leaf = self.experiment(region="west_us", model="llama_3")
        (leaf/".hydra").mkdir()
        (leaf/".hydra"/"config.yaml").write_text(
            "controller:\n  regions:\n  - region_name: west_us\n    model_endpoints:\n    - model_name: llama_3\n")
        self.assertEqual(self.report().metrics.model.iloc[0], "llama_3")

    def test_nonwindow_csv_is_diagnosed(self):
        leaf = self.experiment()
        (leaf/"summary"/"notes.csv").write_text("not a window")
        report = self.report()
        self.assertEqual(report.metrics.n_windows.iloc[0], 1)
        self.assertIn("ignored_filename", {i["code"] for i in report.metadata["diagnostics"]})

    def test_duplicate_numeric_window_names_are_rejected(self):
        leaf = self.experiment()
        folder = leaf/"detailed"/"westus_A_0"
        (folder/"00.csv").write_bytes((folder/"0.csv").read_bytes())
        with self.assertRaisesRegex(InputError, "same numeric window"):
            self.report()

    def test_missing_column_fails_with_filename(self):
        leaf = self.experiment()
        p = leaf/"detailed"/"westus_A_0"/"0.csv"
        df = pd.read_csv(p).drop(columns="ttft_times")
        df.to_csv(p, index=False)
        with self.assertRaisesRegex(InputError, "ttft_times"):
            self.report()

    def test_corrupt_file_is_not_silently_skipped(self):
        leaf = self.experiment()
        (leaf/"detailed"/"westus_A_0"/"1.csv").write_text("broken")
        with self.assertRaises(InputError):
            self.report()
        report = self.report(allow_partial=True)
        self.assertEqual(report.metrics.n_windows.iloc[0], 2)
        self.assertFalse(report.metrics.request_data_complete.iloc[0])
        self.assertTrue(any(i["severity"] == "error" for i in report.metadata["diagnostics"]))

    def test_missing_timings_remain_unknown(self):
        leaf = self.experiment()
        path = leaf/"detailed"/"westus_A_0"/"0.csv"
        df = pd.read_csv(path)
        df[["ttft_times", "tbt_times", "prompt_sizes"]] = np.nan
        df.to_csv(path, index=False)
        row = self.report().metrics.iloc[0]
        for column in ["viol_ttft", "viol_avg_tbt", "prompt_size_cv", "score_demand", "fss"]:
            self.assertTrue(pd.isna(row[column]), column)
        self.assertEqual(row.n_valid_ttft_times, 0)

    def test_single_sample_is_not_perfect_stability(self):
        leaf = self.experiment()
        self.details(leaf, 0, [4])
        row = self.report().metrics.iloc[0]
        self.assertTrue(pd.isna(row.cv_response))
        self.assertTrue(pd.isna(row.prompt_size_cv))
        self.assertTrue(pd.isna(row.score_stability))
        self.assertEqual(row.latency_sample_status, "single_sample")

    def test_invalid_numeric_values_fail_or_become_unknown(self):
        leaf = self.experiment()
        path = leaf/"detailed"/"westus_A_0"/"0.csv"
        for value in [-1, np.inf, "nonsense"]:
            with self.subTest(value=value):
                self.details(leaf, 0, [4, 6, 8])
                df = pd.read_csv(path)
                df["response_times"] = df.response_times.astype(object)
                df.loc[0, "response_times"] = value
                df.to_csv(path, index=False)
                with self.assertRaises(InputError):
                    self.report()
                row = self.report(allow_partial=True).metrics.iloc[0]
                self.assertEqual(row.n_valid_response_times, 2)
                self.assertFalse(row.request_data_complete)

    def test_timing_relationships_and_tbt_formula_are_validated(self):
        leaf = self.experiment()
        path = leaf/"detailed"/"westus_A_0"/"0.csv"
        for column, value in [("queue_times", 100), ("ttft_times", 100), ("tbt_times", 100), ("token_sizes", 0)]:
            self.details(leaf, 0, [4, 6, 8])
            df = pd.read_csv(path)
            df.loc[0, column] = value
            df.to_csv(path, index=False)
            with self.subTest(column=column), self.assertRaisesRegex(InputError, "violate"):
                self.report()

    def test_quiet_windows_count_in_denominator(self):
        leaf = self.experiment()
        self.details(leaf, 1, [])
        self.details(leaf, 2, [4, 6, 8])
        row = self.report().metrics.iloc[0]
        self.assertEqual(row.n_windows, 3)
        self.assertEqual(row.n_requests, 6)
        self.assertEqual(row.requests_per_window, 2)
        self.assertEqual(row.n_quiet_windows, 1)
        self.assertGreater(row.completion_window_cv, 0)

    def test_entirely_quiet_model_is_retained(self):
        leaf = self.experiment()
        self.details(leaf, 0, [])
        row = self.report().metrics.iloc[0]
        self.assertEqual(row.n_requests, 0)
        self.assertEqual(row.n_quiet_windows, 1)
        self.assertTrue(pd.isna(row.e2e_p50))
        self.assertTrue(pd.isna(row.viol_ttft))

    def test_missing_export_is_not_a_known_quiet_window(self):
        leaf = self.experiment()
        self.details(leaf, 1, [])
        (leaf/"detailed"/"westus_A_0"/"1.csv").unlink()
        report = self.report()
        self.assertFalse(report.metrics.request_data_complete.iloc[0])
        self.assertTrue(pd.isna(report.windows.completed_requests.iloc[1]))

    def test_memory_identifies_omitted_scaled_applications(self):
        leaf = self.experiment()
        (leaf/"memory").mkdir()
        pd.DataFrame({"instance": ["westus_A_0_0", "westus_A_9_0"]}).to_csv(leaf/"memory"/"0.csv", index=False)
        report = self.report()
        row = report.metrics.iloc[0]
        self.assertFalse(row.request_data_complete)
        self.assertEqual(row.n_unexported_live_applications, 1)
        self.assertEqual(report.metadata["experiments"][0]["unexported_live_applications"]["0"], ["westus_A_9"])

    def test_resource_integration_does_not_invent_a_tail(self):
        leaf = self.experiment()
        self.cost(leaf, 0, [0, 1800, 3600])
        row = self.report().metrics.iloc[0]
        self.assertEqual(row.prod_vm_h, 2)
        self.assertEqual(row.resource_coverage_status, "observed_intervals_only")
        self.assertTrue(pd.isna(row.vm_h_per_window))

    def test_resource_integration_is_independent_of_file_partition(self):
        leaf = self.experiment()
        self.cost(leaf, 0, [0, 1800, 3600])
        whole = self.report().metrics.prod_vm_h.iloc[0]
        self.cost(leaf, 0, [0])
        self.details(leaf, 1, [])
        self.cost(leaf, 1, [1800, 3600])
        split = self.report().metrics.prod_vm_h.iloc[0]
        self.assertEqual(whole, split)
        self.assertEqual(split, 2)

    def test_single_snapshot_has_unknown_duration(self):
        leaf = self.experiment()
        self.cost(leaf, 0, [100])
        row = self.report().metrics.iloc[0]
        self.assertTrue(pd.isna(row.prod_vm_h))
        self.assertEqual(row.resource_coverage_status, "insufficient_snapshots")

    def test_explicit_boundaries_carry_state_between_windows(self):
        leaf = self.experiment()
        self.details(leaf, 1, [])
        self.cost(leaf, 0, [0])
        self.cost(leaf, 1, [1800], prod=3)
        bounds = self.boundaries({leaf: [(0, 0, 1800), (1, 1800, 3600)]})
        row = self.report(window_boundaries=bounds).metrics.iloc[0]
        self.assertEqual(row.prod_vm_h, 2.5)
        self.assertEqual(row.resource_time_coverage, 1)
        self.assertEqual(row.resource_coverage_status, "complete")
        self.assertEqual(row.vm_h_per_window, 1.25)
        self.assertEqual(row.completed_requests_per_second, 3/3600)

    def test_unknown_initial_state_leaves_partial_coverage(self):
        leaf = self.experiment()
        self.cost(leaf, 0, [100, 1800])
        bounds = self.boundaries({leaf: [(0, 0, 3600)]})
        row = self.report(window_boundaries=bounds).metrics.iloc[0]
        self.assertEqual(row.resource_coverage_status, "partial")
        self.assertAlmostEqual(row.resource_time_coverage, 3500/3600)
        self.assertTrue(pd.isna(row.vm_h_per_window))

    def test_boundary_overlaps_are_rejected(self):
        leaf = self.experiment()
        self.details(leaf, 1, [])
        bounds = self.boundaries({leaf: [(0, 0, 3600), (1, 3500, 7200)]})
        with self.assertRaisesRegex(InputError, "Overlapping"):
            self.report(window_boundaries=bounds)

    def test_boundaries_cannot_contradict_observed_timestamps(self):
        leaf = self.experiment()
        self.cost(leaf, 0, [0, 3600])
        bounds = self.boundaries({leaf: [(0, 0, 3500)]})
        with self.assertRaisesRegex(InputError, "outside supplied boundaries"):
            self.report(window_boundaries=bounds)

    def test_weights_are_validated_and_normalized(self):
        for weights in [{"slo": -1}, {"slo": np.nan}, {"slo": np.inf}, {"unknown": 1},
                        {name: 0 for name in normalize_weights()}]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                normalize_weights(weights)
        weights = normalize_weights({"slo": .5})
        self.assertAlmostEqual(sum(weights.values()), 1)
        self.assertAlmostEqual(weights["slo"], .5/1.15)

    def test_missing_minmax_values_do_not_become_neutral(self):
        values = _minmax(pd.Series([2., np.nan, 2.]))
        self.assertEqual(values.iloc[0], .5)
        self.assertTrue(pd.isna(values.iloc[1]))

    def test_fss_compares_peers_and_contributions_sum_to_score(self):
        first = self.experiment()
        self.details(first, 0, [1.2, 1.4, 1.6])
        second = self.experiment(asynchronous="False")
        self.details(second, 0, [40, 60, 80])
        report = self.report(fss_weights={"cost": 0})
        rows = report.metrics
        self.assertTrue(rows.fss.notna().all())
        self.assertEqual(rows.fss_cohort_id.nunique(), 1)
        self.assertTrue(rows.fss.between(0, 100).all())
        contributions = rows[[c for c in rows if c.startswith("contribution_")]].sum(axis=1)
        np.testing.assert_allclose(rows.fss, contributions)
        self.assertGreater(rows[rows.feed_async == "True"].fss.iloc[0], rows[rows.feed_async == "False"].fss.iloc[0])

    def test_one_configuration_and_unrelated_models_are_not_peers(self):
        leaf = self.experiment()
        self.details(leaf, 0, [4, 6, 8], model="B")
        rows = self.report(fss_weights={"cost": 0}).metrics
        self.assertTrue(rows.fss.isna().all())
        self.assertTrue(rows.fss_status.str.contains("insufficient_comparable_peers").all())

    def test_cohort_reference_changes_when_members_change(self):
        self.experiment()
        self.experiment(asynchronous="False")
        first = self.report(fss_weights={"cost": 0}).metrics.fss_cohort_id.iloc[0]
        self.experiment(run=1)
        second = self.report(fss_weights={"cost": 0}).metrics.fss_cohort_id.iloc[0]
        self.assertNotEqual(first, second)

    def test_missing_cost_does_not_receive_a_favorable_score(self):
        self.experiment()
        self.experiment(asynchronous="False")
        rows = self.report().metrics
        self.assertTrue(rows.score_cost.isna().all())
        self.assertTrue(rows.fss.isna().all())
        self.assertTrue(rows.fss_status.str.contains("incomplete_resource_duration").all())

    def test_full_fss_with_resource_coverage_and_disabled_components(self):
        first = self.experiment()
        second = self.experiment(asynchronous="False")
        self.cost(first, 0, [0, 3600], prod=2)
        self.cost(second, 0, [0, 3600], prod=4)
        bounds = self.boundaries({first: [(0, 0, 3600)], second: [(0, 0, 3600)]})
        rows = self.report(window_boundaries=bounds).metrics
        self.assertTrue(rows.fss.notna().all())
        self.assertEqual(set(rows.score_cost), {0, 1})
        np.testing.assert_allclose(rows.fss, rows[[c for c in rows if c.startswith("contribution_")]].sum(axis=1))
        self.assertEqual(rows.fss_cohort_id.nunique(), 1)

    def test_unusable_memory_audit_does_not_certify_complete_requests(self):
        leaf = self.experiment()
        (leaf/"memory").mkdir()
        (leaf/"memory"/"0.csv").write_text("bad_header\n1\n")
        with self.assertRaises(InputError):
            self.report()
        self.assertFalse(self.report(allow_partial=True).metrics.request_data_complete.iloc[0])

    def test_plot_bar_positions_remain_distinct_with_duplicate_labels(self):
        self.experiment()
        self.experiment(asynchronous="False")
        rows = self.report(fss_weights={"cost": 0}).metrics
        rows["experiment_id"], rows["feed_async"] = "identical-label", "True"
        from inference.plots import plot_fss_bar
        def inspect(fig, out_dir, name):
            positions = [bar.get_y()+bar.get_height()/2 for bar in fig.axes[0].patches]
            self.assertEqual(len(set(positions)), 2)
            import matplotlib.pyplot as plt
            plt.close(fig)
            return Path(name)
        with patch("inference.plots._save", side_effect=inspect):
            plot_fss_bar(rows, Path(self.temp.name))

    def cli(self, *extra):
        out = Path(self.temp.name)/"output"/"report.csv"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = main(["--results", str(self.root), "--out", str(out), *extra])
        return code, out

    def test_cli_writes_provenance_and_no_plots_needs_no_matplotlib(self):
        self.experiment()
        original = __import__
        def restricted(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("matplotlib intentionally unavailable")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=restricted):
            code, out = self.cli("--no-plots")
        self.assertEqual(code, 0)
        self.assertTrue(out.exists())
        self.assertTrue(out.with_name("report_windows.csv").exists())
        metadata = json.loads(out.with_suffix(".metadata.json").read_text())
        self.assertTrue(metadata["input_files"])
        self.assertEqual(metadata["plot_status"], "skipped")
        self.assertFalse(metadata["forecast_provenance_verified"])

    def test_cli_plot_failure_is_partial_success(self):
        self.experiment()
        with patch("inference.plots.generate_all", side_effect=RuntimeError("render failed")):
            code, out = self.cli()
        self.assertEqual(code, 2)
        self.assertTrue(out.exists())
        metadata = json.loads(out.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["plot_status"], "failed")
        self.assertIn("render failed", metadata["plot_error"])

    def test_cli_partial_inputs_return_two(self):
        leaf = self.experiment()
        (leaf/"detailed"/"westus_A_0"/"1.csv").write_text("broken")
        code, _ = self.cli("--allow-partial", "--no-plots")
        self.assertEqual(code, 2)

    def test_cli_invalid_weights_return_one(self):
        self.experiment()
        code, out = self.cli("--slo-weight", "-1", "--no-plots")
        self.assertEqual(code, 1)
        self.assertFalse(out.exists())


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        pd.DataFrame({"arrival_timestamp": [0, 60, 120], "prompt_size": [10, 20, 30],
                      "model_type": ["A"]*3, "regions": ["012"]*3}).to_csv(self.path/"trace.csv", index=False)
        self.prediction = self.path/"forecast_A_region0.csv"
        pd.DataFrame({"Time": [0, 60, 120, 180], "Predicted": [10, 22, 25, 99999]}).to_csv(self.prediction, index=False)

    def evaluate(self):
        return evaluate_forecasts(self.path/"trace.csv", self.path, observation_end=180, evaluation_start=0)

    def test_alignment_error_metrics_and_future_exclusion(self):
        metrics, buckets, metadata = self.evaluate()
        row = metrics.iloc[0]
        self.assertEqual(row.n_scored_buckets, 3)
        self.assertAlmostEqual(row.mae, 7/3)
        self.assertAlmostEqual(row.rmse, np.sqrt(29/3))
        self.assertAlmostEqual(row.wape, 7/60)
        self.assertAlmostEqual(row.mae_skill_vs_persistence, .65)
        self.assertEqual(list(buckets.Time), [0, 60, 120])
        self.assertFalse(metadata["out_of_sample_verified"])

    def test_missing_prediction_is_not_zero(self):
        pd.DataFrame({"Time": [0, 120], "Predicted": [10, 25]}).to_csv(self.prediction, index=False)
        metrics, buckets, _ = self.evaluate()
        self.assertAlmostEqual(metrics.prediction_coverage.iloc[0], 2/3)
        self.assertTrue(pd.isna(buckets.Predicted.iloc[1]))

    def test_invalid_time_grid_and_duplicate_targets_are_rejected(self):
        for times in [[0, 0], [0, 61]]:
            pd.DataFrame({"Time": times, "Predicted": [10, 20]}).to_csv(self.prediction, index=False)
            with self.subTest(times=times), self.assertRaisesRegex(ValueError, "unique minute"):
                self.evaluate()

    def test_partial_observation_buckets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "minute boundaries"):
            evaluate_forecasts(self.path/"trace.csv", self.path, observation_end=179, evaluation_start=0)

    def test_missing_forecast_series_has_zero_coverage_not_zero_error(self):
        trace_path = self.path/"trace.csv"
        trace = pd.read_csv(trace_path, dtype={"regions": str})
        trace.loc[0, "model_type"] = "B"
        trace.to_csv(trace_path, index=False)
        metrics, _, _ = self.evaluate()
        row = metrics[metrics.model == "B"].iloc[0]
        self.assertEqual(row.prediction_coverage, 0)
        self.assertTrue(pd.isna(row.mae))

    def test_zero_actual_demand_has_unknown_wape(self):
        trace_path = self.path/"trace.csv"
        trace = pd.read_csv(trace_path, dtype={"regions": str})
        trace["prompt_size"] = 0
        trace.to_csv(trace_path, index=False)
        metrics, _, _ = self.evaluate()
        self.assertTrue(pd.isna(metrics.wape.iloc[0]))


if __name__ == "__main__":
    unittest.main()
