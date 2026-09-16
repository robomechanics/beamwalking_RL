# Narrow-stance specialist data set

Trot and walk policies commanded over stance widths of 0.05-0.30 m. How they
were trained and evaluated, the file layout and the column meanings are
described in `PIPELINE.md` at the repository root. The CSVs and figures in this
folder are written by the export step when the run completes.

## Summary for review

**Controller.** One trot and one walk policy (Unitree Go2, PPO, 1,800 updates,
seed 5, self-collision on, no domain randomization), trained **without external
disturbances**. Checkpoint hashes are in `policies/`.

**Why not disturbance-trained policies.** Fine-tuning under random pushes was tried
for the reported controller. After it, the trot policy held its front-right leg in stance for
about 70% of the cycle at every commanded duty factor (0.50, 0.625 and 0.75),
width, speed and period, so almost no trot trial met the per-leg duty-factor
and contact-sequence checks. Before that
fine-tuning, all four legs track duty factors 0.55-0.75 within about 0.03 (at 0.50
the rear-right leg runs 0.05-0.08 long). Disturbances are therefore applied only
at evaluation, where they measure robustness.

**What was evaluated**

| Measurement | Scope | Result |
|---|---|---|
| Command fidelity | trot 0.36/0.40/0.48/0.54 s and walk 0.40/0.48/0.54 s, 6 widths, trot duty 0.50-0.75 and walk 0.75-0.90 in 0.05 steps, 64 trials per cell, 0.30 m/s | realized duty rises with the command: 91-100% of the commanded spread for trot, 58-76% for walk (walk at 0.40 s is below the 60% check, where two-tick swings round the schedule) |
| Robustness | the same widths and duty factors at every speed (0.25-0.40 m/s) and period, random base disturbances up to 25 N, 64 trials per cell | success rises with duty factor at narrow stance for both gaits (figure 3, at 0.30 m/s); the stronger 50 N level was not needed because wider stances did not reach full success; per-run counts in `trials/robustness_<gait>_v<speed>_p<period>.json` |
| Period sweep | 4 speeds, 6 widths, every period, 32 trials per cell | periodicity and cost trends in figures 1-2 |
| Duty-factor selector network | 168 contexts (gait, speed, period, width); each labelled with the lowest duty factor keeping at least 90% of the best success rate under disturbance in that context | 133 contexts labelled (trot 76, walk 57) and reproduced exactly by the network; the duty factor it chooses rises as stance narrows (figure 5); 35 contexts have no label, all at 0.05-0.10 m; files in `selector_fit/` and `selector_promoted/` |
| Selector validation | every selection run again under the same disturbances on the held-out test split, 64 trials per context | no test finds a significant shortfall against 90% of the labelled best success rate, in any of the 133 contexts or the 10 gait-and-width groups, so the selector is promoted. 22 of 133 fresh rates sit below their required rate without being significant (smallest p 0.031 against a 3.5e-4 threshold); the per-context test is weak where success is low, which is why the pooled test exists (`selector_promoted/`) |

**Caveats**

- One training seed per gait.
- Figure 5 covers 0.10 m only partially: 8 of 12 trot contexts and 9 of 12 walk
  contexts are labelled there, and the rest are contexts whose best duty factor
  succeeds in under 5% of trials. At 0.05 m no context is labelled. The 0.10 m
  average therefore describes the contexts where some duty factor still works.
- Figure 5 depends on the 90% share. With 80% the average selection is trot
  0.706 at 0.10 m down to 0.50 at 0.25-0.30 m, and with 95% it is 0.725 down to
  0.50; walk falls from 0.822 to 0.75 and from 0.839 to 0.767. The rise as stance
  narrows holds for both gaits either way.
- Within one speed and period, the trot selection never falls as stance narrows
  (0 of 60 neighbouring width pairs). Walk breaks that in 2 of 45 pairs, both at
  0.30 m, where 3 of 12 contexts select 0.80 and lift the average from 0.754 to
  0.762.
- The network reproduces its labels exactly, so figure 5 shows the labels
  themselves. Its evidence is that the rule gives the same end-to-end rise at
  every speed, and that the selections hold on fresh seeds.
- The selection prefers the lower of nearly equally robust duty factors, which
  assumes a less conservative gait is better when robustness is equal.
- The per-context fresh-seed test has little power where success rates are low;
  the gait-and-width test, which pools a width's contexts, is what covers
  0.10 m.
- Periodicity (cycle RMS) is a diagnostic of gait execution, not the convergence
  metric χ of Investigation 1.
- Simulation on flat ground with commanded stance width; no physical narrow
  support.

## Duty factors

| | Trot | Walk |
|---|---|---|
| Training anchors | 0.50, 0.625, 0.75, with duty drawn continuously over 0.50-0.75 | 0.75, 0.80, 0.85, 0.90, with duty drawn continuously over 0.75-0.90 |
| Evaluated | 0.50, 0.55, 0.60, 0.65, 0.70, 0.75 (0.75 from 0.40 s) | 0.75, 0.80, 0.85, 0.90 |

The only shared value is 0.75. Discrete swing ticks round each commanded
schedule, so realized duty factors are unevenly spaced and change with period.

**How to analyze with it**

- Use the realized duty factor, `achieved_df`, as a continuous variable
  instead of the command labels.
- Compare trot and walk at matched realized duty factor, around 0.75 where the
  two ranges meet, as Investigation 1 does.
- Before claiming a difference between neighbouring duty factors, check that
  their `achieved_df` distributions actually separate.

## Figures and the claims they support

Figures are in `figures/`, each with the table it plots. Numbers below are read
from those tables.

| Figure | Paper claim | What it shows |
|---|---|---|
| 1. Periodicity error against realized duty factor | Higher duty factor improves convergence (Investigation 1) | Trot cycle RMS falls as duty factor rises at every stance width, most at narrow stance: 0.107 at duty 0.50 to 0.003 at 0.75 at 0.05 m, and 0.010 to 0.002 at 0.15 m. Walk stays between 0.003 and 0.019 across 0.75–0.90. Periodicity is a diagnostic of gait execution, not the convergence metric χ. |
| 2. Cost of transport against realized duty factor | Duty factor affects energetic cost | Cost of transport (positive mechanical) rises with duty factor for both gaits at every width: trot 0.16 to 0.28 and walk 0.31 to 0.38 at 0.05 m. A higher duty factor costs energy, so it is worth raising where stance is constrained. |
| 3. Robustness (success rate) against stance width | Higher duty factor gives more robust locomotion on narrow terrain (Investigation 3) | At 0.15 m, trot success rises from 32% at duty 0.50 to 74% at 0.75, and walk from 51% at 0.75 to 77% at 0.85. At 0.20 m, trot rises from 90% to 98% and walk from 88% to 99%. Every duty factor fails at 0.05 m and succeeds at 0.25–0.30 m, so duty factor matters where stance is constrained. |
| 4. Realized duty factor by width | The learned controller executes the commanded duty factor (Investigation 2, command fidelity) | Trot realizes 0.52–0.75 for commands 0.50–0.75 and walk 0.76–0.87 for 0.75–0.90, the same at every stance width. |
| 5. Duty factor chosen by the selector network against stance width | As stance width decreases, the selector chooses higher duty factor gaits: duty factor is the parameter to raise when stance is constrained, and not when it is not | Averaged over its speed and period contexts, the network chooses trot 0.50 at 0.25–0.30 m, 0.53 at 0.20 m, 0.68 at 0.15 m and 0.73 at 0.10 m, and walk 0.75–0.76 at 0.25–0.30 m, 0.77 at 0.20 m, 0.80 at 0.15 m and 0.84 at 0.10 m. Each of the four speeds rises end to end on its own (trot 0.50 at 0.30 m to 0.70–0.75 at 0.10 m; walk 0.75–0.78 to 0.80–0.85). Across contexts the choices span 0.65–0.75 for trot and 0.80–0.90 for walk at 0.10 m, and collapse to a single value at 0.25–0.30 m (trot 0.50; walk 0.75, with 0.80 in one to three contexts). The figure draws that full spread faintly behind the interquartile bar. At 0.05 m no context is labelled: 25 of its 28 contexts have no success at all and the other three have one success in 64 trials, below the 5% floor. Figure 5 pools the periods at which every duty factor ran, so its trot contexts exclude 0.36 s. |
