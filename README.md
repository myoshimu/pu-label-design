# Simulation code for *The Advantage of Label Design in Dropout Prediction*

This repository contains the code for the simulation study (Section 7) of

> Katsuragi, M. and Tanaka, K. *The Advantage of Label Design in Dropout
> Prediction: Enhancing Performance through Event-Type Separation.*
> (under review)

The simulation asks two questions that the single-cohort result cannot answer
on its own:

1. **How large is the purification gain, as a function of what is bundled?**
   The paper measures $+0.109$ ROC-AUC on one cohort. The simulation traces
   that gain as a function of $\Delta$, the difference in intrinsic
   predictability between the two bundled events, and of $\rho$, the
   correlation between their risk drivers.
2. **What do the three diagnostics actually detect?** The composition test and
   the $c(x)$ test are shown to answer two *different* questions: whether
   purification pays, and whether SCAR is violated. A label can pass the SCAR
   check and still cost $+0.129$.

## What is and is not included

The design of Section 7 holds the **real feature matrix fixed** and synthesises
**only the labels**. The feature matrix of the study cohort is not distributed
with this release, so the code ships with two ways to run:

| | what it reproduces |
|---|---|
| `exp23_fig_simulation.py` on the bundled `exp22_*.csv` | **Figures 11 and 12 of the paper, exactly.** The CSVs are the output of the simulation run on the real feature matrix. |
| `exp22_simulation.py` from scratch | The **qualitative** result — the shape of the gain surface and the division of labour between the diagnostics — using a surrogate feature matrix (`simlib.surrogate_features`). Individual numbers will not match the paper, because they depend on the real $X$. |

The surrogate matrix has the same number of rows (2,437) and columns (30
numeric + 11 categorical) as the study cohort, and imitates one structural
property that matters for the simulation: the numeric columns are generated
from a common-factor model, because in the real data the attendance, report
and correspondence blocks are each internally redundant. Nothing else about
the real data is encoded in it. The column *names* are the real ones, so that
the code reads against the feature blocks described in Section 3 of the paper
(attendance, report submission, correspondence with the homeroom teacher,
clubs, survey items and enrolment attributes).

## Layout

```
exp22_simulation.py      the simulation itself; writes the three CSVs
exp23_fig_simulation.py  reads the three CSVs; writes Figures 11 and 12
simlib.py                out-of-fold prediction, feature names, surrogate features
pu_style.py              figure styling (colours, fonts)
thesis_style.mplstyle    matplotlib style sheet used for the paper's figures
exp22_sweep.csv          simulation output, real feature matrix  (sweeps A and B)
exp22_diag.csv           simulation output, real feature matrix  (diagnostic 2x2)
exp22_window.csv         simulation output, real feature matrix  (window length)
reference_figures/       Figures 11 and 12 as published, for comparison
LICENSE                  MIT
```

## Running it

```bash
pip install -r requirements.txt

# Reproduce the published figures from the bundled results (seconds)
python exp23_fig_simulation.py

# Re-run the simulation on the surrogate feature matrix (~25 min on 4 cores)
python exp22_simulation.py
```

`exp22_simulation.py` overwrites the three CSVs in place, so copy them first if
you want to keep the published results alongside your own run.

## The generative model

For each replicate, two latent risks are drawn as sparse random linear
combinations of the standardised numeric features $z(x)$:

$$\eta_A = w_A^\top z, \qquad
  \eta_B = \rho\,\eta_A + \sqrt{1-\rho^2}\,\eta_\perp$$

Each event is then Bernoulli with
$P(k=1\mid x) = \sigma(a_k + b_k \eta_k)$, where $(a_k, b_k)$ are solved
numerically to hit a target pair (attainable Bayes ROC-AUC, prevalence). The
observed labels are $s_k = k \cdot \mathrm{Bern}(c_k)$ and the bundled label is
$s_{\text{mix}} = s_A \vee s_B$.

The point of this construction is that **SCAR holds exactly within each
component** — $c_k$ does not depend on $x$. Any apparent SCAR violation
measured on the bundled label is therefore produced solely by the two
components having different $c$, which is the generative-model version of the
paper's central claim.

Constants taken from the study cohort (`EMP` in `exp22_simulation.py`):

| quantity | value |
|---|---|
| prevalence of the bundle | $(196+336)/2437 = 0.218$ |
| share of the bundle from the less predictable event | $336/532 = 0.632$ |
| label frequencies | $c_A = 0.990$, $c_B = 0.458$ |
| attainable ROC-AUC | $0.868$ and $0.664$, so $\Delta = 0.204$ |

Two constants are fitted rather than measured. `SHRINK = 0.04` is the gap
between attainable and finite-sample ROC-AUC, calibrated on a preliminary
sweep. `RHO_FIT = 0.8` is chosen so that the realised bundled-label ROC-AUC
matches the observed $0.750$; it is the only free parameter of the design that
cannot be measured directly.

## Notes

- Source comments are in Japanese; this README and all figure labels are in
  English.
- All randomness is seeded (`SEED = 0`, and per-replicate seeds derived from
  the replicate index), so runs are reproducible on a given package set.
- The code touches no database and reads no institutional data.

## Licence

MIT. See [`LICENSE`](LICENSE).

## Citation

TODO: fill in once the paper has a DOI.
