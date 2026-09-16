# Narrow-stance duty-factor study

Code and data behind the paper's RL results: one trot and one walk narrow-stance
specialist for the Unitree Go2, their evaluation over stance width, speed, gait
period and duty factor, a duty-factor selector network fit on success under
disturbance, and the five paper figures.

`PIPELINE.md` is the method: what each stage runs, how the selector is labelled
and validated, and what every figure shows.

## Layout

| Path | Contents |
|---|---|
| `scripts/pipeline.sh` | the whole run, stages 1-8, resumable |
| `scripts/` | training, evaluation, period sweep, selector and figure entry points |
| `source/beam_walking/beam_walking/experiment/` | task and environment, command sampler, gait schedule, disturbances, trial analysis, selector model |
| `tests/` | unit tests; they run without Isaac Sim |
| `paper_data/specialist_narrow_20260914/` | the exported data set: per-trial CSVs, robustness summaries, the selector fit and its validation, policy provenance, and the figures with their tables |

`results/` holds every run's archives and is not tracked.

## Install

Isaac Lab 5.1 and its Python, then:

```bash
git lfs pull                      # the selector checkpoints in paper_data/ are LFS objects
python -m pip install -e source/beam_walking
```

## Run

```bash
mkdir -p results
setsid nohup bash scripts/pipeline.sh > results/pipeline.log 2>&1 &
tail -f results/narrow_20260914.status
```

Completed stages are skipped on a restart. `TAG` starts a separate run,
`PYTHON_BIN` points at the Isaac Lab Python, `WORKERS` sets how many evaluations
run at once, and `REPO_ROOT` is needed when running a copy of the script from
another directory.

```bash
python -m pytest tests -q
```

Evaluation refuses a checkpoint whose training sources no longer hash to the
values recorded at training time, so the files listed in
`NARROW_TRAINING_SOURCE_FILES` and `NARROW_TASK_FILES` (`narrow_specialist.py`)
must stay byte-identical for the published policies to remain usable. For the
same reason `scripts/beam_experiment.py` is kept although this pipeline never
runs it: the evaluator hashes it as part of the training lineage.
