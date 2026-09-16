"""Fit the narrow-stance duty-factor selector from success under disturbance.

Contexts are the gait, speed, period and stance width of every robustness run
of a tag (results/push_robustness_narrow_<gait>_v<speed>_p<period>_<tag>,
disturbances up to 25 N). A context's label is the lowest commanded duty factor
whose success rate is at least 90% of the best success rate any duty factor
reaches in that context. A context whose best success rate is below 5% has no
label. The classifier (UnifiedDutySelector) must reproduce every label exactly.
Fresh-seed validation and promotion follow in validate_robust_duty_selector.py.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment import unified_duty_selector as selector
from beam_walking.experiment.duty_selector import selector_state_sha256
from beam_walking.experiment.protocol import GAITS, SPEED_RANGE

SCHEMA = "robust_duty_selector_v1"
RETAINED_SUCCESS = Fraction(9, 10)
MINIMUM_BEST_SUCCESS = Fraction(1, 20)
DISTURBANCE_FORCE_N = 25.
BOOTSTRAP = 100_000
EPOCH_SCHEDULE = (3000, 10000, 30000)
RUN = re.compile(r"^push_robustness_narrow_(trot|walk)_v(\d{3})_p(\d\.\d\d)_(.+)$")
CONTEXT = ("gait", "speed", "period", "step_width")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label(counts):
    """counts: duty -> (successes, trials). Lowest duty keeping RETAINED_SUCCESS of the best rate,
    or None when the best rate is below MINIMUM_BEST_SUCCESS."""
    rates = {duty: Fraction(s, n) for duty, (s, n) in counts.items()}
    best = max(rates.values())
    if best < MINIMUM_BEST_SUCCESS or not best:
        return None
    return min(duty for duty, rate in rates.items() if rate >= RETAINED_SUCCESS * best)


def label_interval(counts, rng):
    """2.5th-97.5th percentile of the label over binomial resamples of every duty factor's trials."""
    duties = sorted(counts)
    k = np.array([counts[d][0] for d in duties])
    n = np.array([counts[d][1] for d in duties])
    s = rng.binomial(n, k / n, size=(BOOTSTRAP, len(duties)))
    best = (s / n).argmax(axis=1)
    s_best, n_best = s[np.arange(BOOTSTRAP), best], n[best]
    keeps = (RETAINED_SUCCESS.denominator * s * n_best[:, None]
             >= RETAINED_SUCCESS.numerator * s_best[:, None] * n)
    labelled = (MINIMUM_BEST_SUCCESS.denominator * s_best >= MINIMUM_BEST_SUCCESS.numerator * n_best) & (s_best > 0)
    draws = np.asarray(duties)[keeps.argmax(axis=1)][labelled]
    if not draws.size:
        return None, None
    low, high = np.quantile(draws, [.025, .975], method="inverted_cdf")
    return float(low), float(high)


def load_runs(results, tag):
    """{(gait, speed, period): run} for every complete 25 N robustness run of the tag."""
    runs = {}
    for directory in sorted(results.iterdir()):
        match = RUN.match(directory.name)
        if not match or match.group(4) != tag or not directory.is_dir():
            continue
        gait, speed, period = match.group(1), int(match.group(2)) / 100, float(match.group(3))
        paths = {name: directory / name for name in
                 ("evaluation_complete.json", "evaluation_manifest.json", "perturbation_summary.json")}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"Incomplete robustness run: {directory}")
        completion = json.loads(paths["evaluation_complete.json"].read_text())
        manifest = json.loads(paths["evaluation_manifest.json"].read_text())
        summary = json.loads(paths["perturbation_summary.json"].read_text())
        if (completion.get("complete") is not True or completion.get("external_pushes") is not True
                or manifest.get("split") != "validation" or manifest.get("gait") != gait
                or abs(manifest.get("speed") - speed) > 1e-6 or abs(manifest.get("period") - period) > 1e-6
                or float(summary["perturbation"]["max_force_n"]) != DISTURBANCE_FORCE_N
                or completion.get("checkpoint_sha256") != manifest.get("checkpoint_sha256")):
            raise ValueError(f"Not a complete {DISTURBANCE_FORCE_N:g} N robustness run: {directory}")
        counts = {}
        for c in summary["conditions"].values():
            if c["trials"]:
                counts[(round(c["step_width"], 2), round(c["df"], 3))] = (int(c["success"]), int(c["trials"]))
        runs[(gait, round(speed, 2), round(period, 2))] = {
            "directory": directory, "checkpoint_sha256": completion["checkpoint_sha256"], "counts": counts,
            "max_torque_nm": float(summary["perturbation"]["max_torque_nm"]),
            "sha256": {name: sha256(path) for name, path in paths.items()}}
    return runs


def check_grid(runs, speeds, widths, periods):
    """The runs must be exactly the grid: every speed and period of each gait, one checkpoint per
    gait, and at every period all widths crossed with the gait's duty factors feasible there."""
    for gait in GAITS:
        expected = {(gait, round(v, 2), round(p, 2)) for v in speeds for p in periods[gait]}
        present = {key for key in runs if key[0] == gait}
        if present != expected:
            raise ValueError(f"{gait} robustness runs differ from the grid: missing {sorted(expected - present)}, "
                             f"unexpected {sorted(present - expected)}")
        if len({runs[key]["checkpoint_sha256"] for key in present}) != 1:
            raise ValueError(f"{gait} robustness runs use different checkpoints")
        for key in sorted(present):
            conditions = {(round(w, 2), round(d, 3)) for w in widths for d in selector.feasible_levels(gait, key[2])}
            if set(runs[key]["counts"]) != conditions:
                raise ValueError(f"{key} conditions differ from the grid")
    if len({run["max_torque_nm"] for run in runs.values()}) != 1:
        raise ValueError("Robustness runs use different disturbance torques")


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gait_lists(items, kind):
    """['trot=.36,.40', 'walk=.40'] -> {'trot': [.36, .40], 'walk': [.40]}"""
    out = {}
    for item in items:
        gait, _, values = item.partition("=")
        if gait not in GAITS or not values:
            raise argparse.ArgumentTypeError(f"--{kind} expects GAIT=V,V,... for each gait, got {item}")
        out[gait] = sorted(float(value) for value in values.split(","))
    if set(out) != set(GAITS):
        raise argparse.ArgumentTypeError(f"--{kind} needs every gait")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speeds", type=float, nargs="+", help="grid speeds, m/s")
    parser.add_argument("--widths", type=float, nargs="+", help="grid stance widths, m")
    parser.add_argument("--periods", nargs="+", help="grid periods per gait: trot=.36,.40 walk=.40")
    parser.add_argument("--duties", nargs="+", help="evaluated duty factors per gait: trot=.50,.55 walk=.75")
    parser.add_argument("--exploratory", action="store_true",
                        help="accept runs without the grid arguments (testing only; such a selector cannot be promoted)")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    if not args.exploratory and not all((args.speeds, args.widths, args.periods, args.duties)):
        parser.error("--speeds, --widths, --periods and --duties define the grid a primary fit requires")
    runs = load_runs(args.results, args.tag)
    if not runs:
        raise ValueError("No robustness runs found")
    widths = sorted({w for run in runs.values() for w, _ in run["counts"]})
    selector.set_width_range((widths[0], widths[-1]))
    selector.set_gait_levels(gait_lists(args.duties, "duties") if args.duties else
                             {gait: sorted({d for key, run in runs.items() if key[0] == gait
                                            for _, d in run["counts"]}) for gait in GAITS})
    if not args.exploratory:
        check_grid(runs, args.speeds, args.widths, gait_lists(args.periods, "periods"))
    args.output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    candidates, targets, rejected = [], [], []
    for (gait, speed, period), run in sorted(runs.items()):
        for width in sorted({w for w, _ in run["counts"]}):
            counts = {d: run["counts"][(w, d)] for w, d in run["counts"] if w == width}
            context = {"gait": gait, "speed": speed, "period": period, "step_width": width}
            best_s, best_n = max(counts.values(), key=lambda sn: Fraction(*sn))
            chosen = label(counts)
            for d, (s, n) in sorted(counts.items()):
                candidates.append({**context, "command_df": d, "success": s, "trials": n,
                                   "success_rate": s / n, "labelled": int(d == chosen)})
            if chosen is None:
                rejected.append({**context, "best_success": best_s, "best_trials": best_n,
                                 "reason": "no duty factor succeeds" if best_s == 0
                                 else "best success rate below 5%"})
                continue
            low, high = label_interval(counts, rng)
            s, n = counts[chosen]
            targets.append({**context, "target_df": chosen, "target_success": s, "target_trials": n,
                            "best_success": best_s, "best_trials": best_n,
                            "target_df_ci_low": low, "target_df_ci_high": high,
                            "candidates": " ".join(f"{d:g}" for d in sorted(counts))})
    if len(targets) < 2:
        raise ValueError("Too few labelled contexts to fit the selector")

    for epochs in EPOCH_SCHEDULE:
        model, losses = selector.fit_selector(targets, seed=0, epochs=epochs)
        predictions = selector.predict_rows(model, targets)
        if all(abs(row["selected_df"] - row["target_df"]) < 1e-6 for row in predictions):
            break
    else:
        raise RuntimeError("Classifier did not reproduce every label")

    write_rows(args.output / "candidate_success.csv", candidates)
    write_rows(args.output / "selector_targets.csv", targets)
    if rejected:
        write_rows(args.output / "rejected_contexts.csv", rejected)
    write_rows(args.output / "selector_predictions.csv",
               [{key: row[key] for key in (*CONTEXT, "target_df", "selected_df", "target_df_ci_low",
                                           "target_df_ci_high")} for row in predictions])
    checkpoints = {gait: sorted({run["checkpoint_sha256"] for key, run in runs.items() if key[0] == gait})
                   for gait in GAITS}
    payload = {
        "schema": SCHEMA,
        "state_dict": model.state_dict(), "hidden_dims": [32, 32],
        "input_fields": ["step_width", "speed", "period", "gait_id"],
        "input_ranges": {"step_width": list(selector.WIDTH_RANGE), "speed": list(SPEED_RANGE),
                         "period": list(selector.PERIOD_RANGE), "gait_id": [0, len(GAITS) - 1]},
        "gait_levels": {gait: list(levels) for gait, levels in selector.GAIT_LEVELS.items()},
        "label_rule": ("lowest commanded duty factor whose success rate under disturbance is at least "
                       "90% of the best success rate any duty factor reaches in the context; no label when "
                       "the best success rate is below 5%"),
        "retained_success": [RETAINED_SUCCESS.numerator, RETAINED_SUCCESS.denominator],
        "minimum_best_success": [MINIMUM_BEST_SUCCESS.numerator, MINIMUM_BEST_SUCCESS.denominator],
        "disturbance_max_force_n": DISTURBANCE_FORCE_N,
        "disturbance_max_torque_nm": sorted({run["max_torque_nm"] for run in runs.values()})[0],
        "grid": {"speeds": args.speeds, "widths": args.widths,
                 "periods": gait_lists(args.periods, "periods") if args.periods else None},
        "supported_contexts": [{key: row[key] for key in (*CONTEXT, "target_df", "target_success",
                                                          "target_trials", "best_success", "best_trials")}
                               for row in targets],
        "rejected_contexts": rejected,
        "robustness_runs": [{"gait": g, "speed": v, "period": p, "directory": run["directory"].name,
                             "checkpoint_sha256": run["checkpoint_sha256"], "sha256": run["sha256"]}
                            for (g, v, p), run in sorted(runs.items())],
        "checkpoint_sha256": {gait: shas[0] for gait, shas in checkpoints.items() if len(shas) == 1},
        "fit_seed": 0, "epochs": epochs, "final_cross_entropy": losses[-1],
        "deployment_ready": False, "exploratory": args.exploratory,
        "selector_runtime_sha256": selector.selector_runtime_sha256(ROOT),
        "fit_script_sha256": sha256(Path(__file__)),
        "selector_state_sha256": selector_state_sha256(model.state_dict()),
    }
    torch.save(payload, args.output / "robust_duty_selector.pt")
    report = {
        "schema": "robust_duty_selector_fit_v1",
        "contexts": len(targets), "rejected_contexts": len(rejected),
        "contexts_per_gait": {g: sum(row["gait"] == g for row in targets) for g in GAITS},
        "selected_levels_per_gait": {g: sorted({row["selected_df"] for row in predictions if row["gait"] == g})
                                     for g in GAITS},
        "epochs": epochs, "final_cross_entropy": losses[-1], "exploratory": args.exploratory,
        "selector_checkpoint_sha256": sha256(args.output / "robust_duty_selector.pt"),
    }
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2))
    print("ROBUST_DUTY_SELECTOR_FIT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
