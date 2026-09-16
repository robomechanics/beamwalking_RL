"""Validate and promote the robust duty-factor selector on fresh seeds.

For every supported context, the selector's duty factor is run again under the
same disturbances on the held-out test split
(results/narrow_robust_selector_validation_<gait>_v<speed>_p<period>_<tag>, one
run per gait, speed and period over the context widths and the selected duty
factors). Each context's required success rate is 90% of the best success rate
in its labelling runs. Two one-sided exact tests, Bonferroni corrected together
to a family-wise 5%, look for fresh success significantly below that rate: one
per context, and one per gait and stance width over all of its contexts, which
keeps power where success rates are low. When no test fails, the selector is
saved next to the evidence as validated_robust_duty_selector.pt with
deployment_ready=True.

--print-jobs lists the validation runs for the pipeline instead.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment import unified_duty_selector as selector  # noqa: E402
from beam_walking.experiment.duty_selector import selector_state_sha256  # noqa: E402

SCHEMA = "robust_duty_selector_v1"
FAMILY_ALPHA = .05
VALIDATION_SEED = 1000000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Not a robust duty-factor selector")
    selector.set_width_range(payload["input_ranges"]["step_width"])
    selector.set_gait_levels(payload["gait_levels"])
    if payload["selector_runtime_sha256"] != selector.selector_runtime_sha256(ROOT):
        raise ValueError("Selector runtime source differs from the fit")
    model = selector.UnifiedDutySelector(tuple(payload["hidden_dims"]))
    model.load_state_dict(payload["state_dict"])
    if selector_state_sha256(model.state_dict()) != payload["selector_state_sha256"]:
        raise ValueError("Selector parameter hash mismatch")
    model.eval()
    contexts = payload["supported_contexts"]
    predictions = selector.predict_rows(model, [dict(row) for row in contexts])
    if any(abs(row["selected_df"] - row["target_df"]) > 1e-6 for row in predictions):
        raise ValueError("Selector no longer reproduces its labels")
    return payload, predictions


def groups(predictions):
    """{(gait, speed, period): [prediction rows]}"""
    out = defaultdict(list)
    for row in predictions:
        out[(row["gait"], row["speed"], row["period"])].append(row)
    return dict(sorted(out.items()))


def run_directory(results, tag, gait, speed, period):
    return results / f"narrow_robust_selector_validation_{gait}_v{round(speed * 100):03d}_p{period:.2f}_{tag}"


def binomial_pmf(n, p):
    i = np.arange(n + 1)
    log_comb = np.array([sum(np.log(np.arange(1, n + 1))) - sum(np.log(np.arange(1, k + 1)))
                         - sum(np.log(np.arange(1, n - k + 1))) for k in i])
    return np.exp(log_comb + i * np.log(p) + (n - i) * np.log1p(-p))


def lower_tail(successes, trials_and_rates):
    """P(sum of independent Binomial(n, p) <= successes)."""
    pmf = np.ones(1)
    for n, p in trials_and_rates:
        pmf = np.convolve(pmf, binomial_pmf(n, p))
    return float(min(1., pmf[:successes + 1].sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-jobs", action="store_true")
    args = parser.parse_args()
    payload, predictions = load(args.selector)
    by_run = groups(predictions)
    if args.print_jobs:
        for (gait, speed, period), rows in by_run.items():
            widths = ",".join(f"{w:g}" for w in sorted({row["step_width"] for row in rows}))
            duties = ",".join(f"{d:g}" for d in sorted({row["selected_df"] for row in rows}))
            print(f"selector_validation_job {gait} {speed:g} {period:g} {widths} {duties}")
        return
    if args.output is None:
        parser.error("--output is required")
    if payload.get("exploratory") is not False or payload.get("deployment_ready") is not False:
        parser.error("Only a primary, not-yet-validated selector can be promoted")
    retained = payload["retained_success"][0] / payload["retained_success"][1]
    contexts, evidence = [], []
    for (gait, speed, period), rows in by_run.items():
        directory = run_directory(args.results, args.tag, gait, speed, period)
        paths = {name: directory / name for name in
                 ("evaluation_complete.json", "evaluation_manifest.json", "perturbation_summary.json")}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"Validation run is missing: {directory}")
        completion = json.loads(paths["evaluation_complete.json"].read_text())
        manifest = json.loads(paths["evaluation_manifest.json"].read_text())
        summary = json.loads(paths["perturbation_summary.json"].read_text())
        counts = {(round(c["step_width"], 2), round(c["df"], 3)): (int(c["success"]), int(c["trials"]))
                  for c in summary["conditions"].values()}
        widths = {row["step_width"] for row in rows}
        duties = {row["selected_df"] for row in rows}
        if (completion.get("complete") is not True or completion.get("external_pushes") is not True
                or manifest.get("split") != "test" or manifest.get("seeds", [None])[0] != VALIDATION_SEED
                or manifest.get("gait") != gait or abs(manifest.get("speed") - speed) > 1e-6
                or abs(manifest.get("period") - period) > 1e-6
                or float(summary["perturbation"]["max_force_n"]) != payload["disturbance_max_force_n"]
                or float(summary["perturbation"]["max_torque_nm"]) != payload["disturbance_max_torque_nm"]
                or completion.get("checkpoint_sha256") != payload["checkpoint_sha256"][gait]
                or set(counts) != {(round(w, 2), round(d, 3)) for w in widths for d in duties}):
            raise ValueError(f"Validation run does not match this selector: {directory}")
        evidence.append({"gait": gait, "speed": speed, "period": period, "directory": directory.name,
                         "sha256": {name: sha256(path) for name, path in paths.items()}})
        for row in rows:
            s, n = counts[(round(row["step_width"], 2), round(row["selected_df"], 3))]
            contexts.append({
                "gait": gait, "speed": speed, "period": period, "step_width": row["step_width"],
                "selected_df": row["selected_df"], "fresh_success": s, "fresh_trials": n,
                "fresh_success_rate": s / n,
                "labelled_success_rate": row["target_success"] / row["target_trials"],
                "best_labelled_success_rate": row["best_success"] / row["best_trials"],
                "required_success_rate": retained * row["best_success"] / row["best_trials"]})
    by_width = defaultdict(list)
    for row in contexts:
        by_width[(row["gait"], row["step_width"])].append(row)
    alpha = FAMILY_ALPHA / (len(contexts) + len(by_width))
    for row in contexts:
        row["p_value"] = lower_tail(row["fresh_success"], [(row["fresh_trials"], row["required_success_rate"])])
        row["passed"] = int(row["p_value"] >= alpha)
    widths_summary = []
    for (gait, width), rows in sorted(by_width.items()):
        fresh = sum(row["fresh_success"] for row in rows)
        trials = sum(row["fresh_trials"] for row in rows)
        required = [(row["fresh_trials"], row["required_success_rate"]) for row in rows]
        p_value = lower_tail(fresh, required)
        widths_summary.append({
            "gait": gait, "step_width": width, "contexts": len(rows), "fresh_success": fresh, "fresh_trials": trials,
            "fresh_success_rate": fresh / trials,
            "required_success_rate": sum(n * p for n, p in required) / trials,
            "p_value": p_value, "passed": int(p_value >= alpha)})
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("selector_validation_summary.csv", contexts),
                       ("selector_validation_by_width.csv", widths_summary)):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    failed = [row for row in contexts if not row["passed"]]
    failed_widths = [row for row in widths_summary if not row["passed"]]
    report = {
        "schema": "robust_duty_selector_validation_v1",
        "contexts": len(contexts), "passed_contexts": len(contexts) - len(failed),
        "width_groups": len(widths_summary), "passed_width_groups": len(widths_summary) - len(failed_widths),
        "all_contexts_passed": not failed and not failed_widths,
        "per_test_alpha": alpha, "family_alpha": FAMILY_ALPHA,
        "failed_contexts": [{key: row[key] for key in ("gait", "speed", "period", "step_width", "selected_df",
                                                      "fresh_success", "fresh_trials", "required_success_rate")}
                            for row in failed],
        "failed_width_groups": failed_widths,
        "selector_checkpoint_sha256": sha256(args.selector),
    }
    if report["all_contexts_passed"]:
        validated = dict(payload, deployment_ready=True, validation_evidence=evidence,
                         validation_selector_checkpoint_sha256=sha256(args.selector),
                         validation_split="test", validation_seed_start=VALIDATION_SEED)
        torch.save(validated, args.output / "validated_robust_duty_selector.pt")
        report["validated_selector"] = "validated_robust_duty_selector.pt"
    (args.output / "validation_report.json").write_text(json.dumps(report, indent=2))
    print("ROBUST_DUTY_SELECTOR_VALIDATION", json.dumps(report), flush=True)
    if not report["all_contexts_passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
