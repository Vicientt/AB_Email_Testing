# Hillstrom E‑mail A/B Test & Uplift Modeling

*A short proposal + full analysis & reproducibility guide*

---

## Table of Contents

* Executive summary
* Business questions & success criteria
* Dataset
* Methods

  * Randomization check (EDA)
  * Classical A/B tests
  * Uplift modeling (T-learner RF)
  * ROI simulation (top-k%)
* Decisions & recommendations
* Modeling roadmap (next steps)
* Reproducibility
* Assumptions & caveats
* How to extend

---

## Proposal

### Objective

Optimize the e‑mail targeting policy for **Mens E‑Mail** and **Womens E‑Mail** relative to **No E‑Mail (control)** to **maximize incremental profit** while protecting customer experience (deliverability, complaints, unsubscribes).

### Scope & hypotheses

* **H1:** Mens campaign increases conversion and average spend vs. control.
* **H2:** Womens campaign increases conversion and average spend vs. control.
* **H3:** A **selective, uplift‑ranked policy** (mail top‑*k*%) **outperforms mail‑all** on net profit.

### Data & economic assumptions

* **Dataset:** Hillstrom (2008) with fields:
  `recency, history_segment, history, mens, womens, zip_code, newbie, channel, visit, conversion, spend, segment`.
* **Unit economics (configurable in `roi.py`):** **\$0.10/email** send cost; **\$15 contribution margin per incremental conversion**.

  > Run sensitivity analysis with your own margin/costs before deployment.

### Methodology

1. **Randomization check** — Validate balance of pre‑treatment covariates (e.g., `history`, `recency`) across treatment arms.
2. **Classical A/B tests** — Two‑proportion **z‑tests** for conversion; **Welch’s t‑test** with **5,000 bootstrap** resamples for spend differences.
3. **Uplift modeling** — **T‑learner** with calibrated Random Forests; evaluate **Qini curve/AUC** and **uplift\@k** on a hold‑out split.
4. **Policy search & ROI** — Rank customers by predicted uplift and simulate **net profit** at *k* ∈ {5%, 10%, 20%, 30%, 100%}.
5. **Visualization** — Plot **Qini curve vs. k** and save to `./figures/` (e.g., `figures/qini_mens_e-mail_vs_no_e-mail.png`, `figures/qini_womens_e-mail_vs_no_e-mail.png`).

### Results (current run)

* **Conversion lift**

  * Mens vs Control: z = **7.385**, *p* ≈ **1.52e‑13**; rates **1.253% vs 0.573%**; **+0.681 pp** (≈ **+118.8%** rel.).
  * Womens vs Control: z = **3.780**, *p* ≈ **1.57e‑4**; rates **0.884% vs 0.573%**; **+0.311 pp** (≈ **+54.3%** rel.).
  * Mens vs Womens: z = **3.713**, *p* ≈ **2.05e‑4**; **+0.369 pp** (≈ **+41.8%** rel.) favoring Mens.
* **Spend lift (per customer)**

  * Mens vs Control: *t* = **5.300**, *p* ≈ **1.16e‑7**; **+\$0.770** (95% CI **\[+\$0.479, +\$1.049]**).
  * Womens vs Control: *t* = **3.256**, *p* ≈ **0.00113**; **+\$0.424** (95% CI **\[+\$0.159, +\$0.684]**).
* **Uplift quality**

  * Mens vs Control: **Qini AUC = −0.142** (underperforms random in this split).
  * Womens vs Control: **Qini AUC = +0.094** (modest positive signal).
* **ROI (top‑*k* policy)**

  * **Mens:** Top 10% → **+\$27.72** net; mailing **all** → **+\$11.60**.
  * **Womens:** Generally **negative**; **top 5% ≈ break‑even** (−\$0.03).

### Recommendation & rollout

* **Mens E‑Mail:** Deploy a **top‑\~10% uplift policy** with a persistent **control/holdout** to measure incrementality; monitor deliverability and unsubscribe.
* **Womens E‑Mail:** **Pause** or **restrict to top 5%** while improving features, calibration, and economics via sensitivity tests.
* **Experiment design:** Run a **multicell test** (Mens: top‑10% vs all vs control; Womens: top‑5% vs all vs control).
  **Primary KPI:** incremental profit; **guardrails:** complaint rate, unsubscribe, deliverability.

### Deliverables

* Reproducible pipeline (`main.py`) generating statistical tables, **Qini plots**, and ROI tables.
* PNGs under `./figures/` with Qini curves vs. k.
* One‑page business summary comparing **uplift policy vs mail‑all**.

### Success criteria

* **Qini AUC > 0** on hold‑out; **incremental profit > mail‑all** at chosen *k*; stable results in follow‑up split tests.

---

## Executive summary

We analyze the Hillstrom 2008 e‑mail experiment to answer: **Should we send men’s and women’s marketing e‑mails, and to whom?** Using classical A/B tests plus an uplift model with ROI simulation, we find:

* **Conversion (A/B)**

  * **Mens E‑Mail vs Control:** 1.253% vs 0.573% (Δ = **+0.681 pp**, **+118.8%** rel.), *p* ≈ 1.52e‑13 → **statistically significant**.
  * **Womens E‑Mail vs Control:** 0.884% vs 0.573% (Δ = **+0.311 pp**, **+54.3%** rel.), *p* ≈ 1.57e‑4 → **statistically significant**.
  * **Mens vs Womens:** 1.253% vs 0.884% (Δ = **+0.369 pp**, **+41.8%** rel.), *p* ≈ 2.05e‑4 → **men respond more**.
* **Spend (Welch + bootstrap CI)**

  * **Mens vs Control:** Δ̄ = **+\$0.770** per customer (95% CI **\[+\$0.479, +\$1.049]**), *p* ≈ 1.16e‑7 → **significant lift in spend**.
  * **Womens vs Control:** Δ̄ = **+\$0.424** (95% CI **\[+\$0.159, +\$0.684]**), *p* ≈ 0.00113 → **significant**.
* **Uplift modeling & ROI** *(margin = \$15/conversion, cost = \$0.10/email)*

  * **Mens vs Control:** Qini AUC = **–0.142** (weak model overall), but **targeting top 10%** by predicted uplift yields **+\$27.72** net in this split; mailing everyone gives **+\$11.60**.
  * **Womens vs Control:** Qini AUC = **+0.094** (modest), but **profits are negative** at tested k% except the **top 5% ≈ break‑even** (–\$0.03).

**Recommendation**

1. **Men:** mail a **small, high‑uplift slice (≈10%)** rather than all—higher profit under current assumptions.
2. **Women:** **hold‑out or restrict to the very top tier**, and improve the uplift model before scaling.
3. Validate assumptions (margin, cost, deliverability) and iterate on features/modeling to improve Qini/ROI.

---

---

## Business questions & success criteria

* **Q1.** Do men’s/women’s e‑mails increase conversion and revenue vs. control?
  **Metric:** z‑tests on conversion; Welch’s *t* on spend; decide if *p* < 0.05 and lift is material.
* **Q2.** Can we **selectively target** customers to improve profit?
  **Metric:** Uplift model quality (Qini AUC) and **simulated net profit** at various target rates *k*.

---

---

## Dataset

The Hillstrom e‑mail dataset contains targets **visit**, **conversion**, **spend**; treatment **segment**; and customer covariates.

**Columns observed**

```
['recency', 'history_segment', 'history', 'mens', 'womens', 'zip_code', 'newbie', 'channel', 'visit', 'conversion', 'spend', 'segment']
```

---

---

## Methods

### 1) Randomization check (EDA)

We summarize **baseline variables by segment** to confirm balance across groups (means/SDs). Result shows **similar recency/history** across treatment arms, consistent with random assignment.

**Observed (head)**

```
segment        variable   history    recency
Mens E‑Mail       mean   242.836      5.774
Mens E‑Mail        std   260.356      3.513
No E‑Mail         mean   240.883      5.750
No E‑Mail          std   252.739      3.498
Womens E‑Mail     mean   242.537      5.768
```

### 2) Classical A/B tests

* **Conversion:** two‑sample **z‑test for proportions**; we report z, *p*, absolute/relative lift, and group rates.
* **Spend:** **Welch’s t‑test** plus **bootstrap 95% CI** for the mean difference.

**Key outputs**

* **Mens vs Control (conversion):** z = 7.385, *p* ≈ 1.52e‑13, **abs lift = 0.006805** (0.681 pp), **rel lift = 118.84%**; rates: 0.01253 vs 0.00573.

* **Womens vs Control (conversion):** z = 3.780, *p* ≈ 1.57e‑4, **abs lift = 0.003111** (0.311 pp), **rel lift = 54.33%**; rates: 0.00884 vs 0.00573.

* **Mens vs Womens (conversion):** z = 3.713, *p* ≈ 2.05e‑4, **abs diff = 0.003694** (0.369 pp), **rel = 41.80%**.

* **Mens vs Control (spend):** *t* = 5.300, *p* ≈ 1.16e‑7, **Δ̄ = +\$0.770** (95% CI **\[+\$0.479, +\$1.049]**).

* **Womens vs Control (spend):** *t* = 3.256, *p* ≈ 0.00113, **Δ̄ = +\$0.424** (95% CI **\[+\$0.159, +\$0.684]**).

### 3) Uplift modeling (T‑learner RF)

We build a **T‑learner**: two calibrated Random Forests (treatment/control) on an engineered design matrix (numeric passthrough + one‑hot for categoricals). We compute **uplift = P(y|treat) − P(y|control)** on a hold‑out fold.

* **Mens vs Control:** **Qini AUC = −0.142** (underperforms random overall).
* **Womens vs Control:** **Qini AUC = +0.094** (modest signal).

> *Interpretation:* Negative/low Qini can arise from **limited features**, **calibration variance**, **class imbalance**, and **model capacity**.

### 4) ROI simulation (target top‑k%)

Assuming **margin = \$15** per incremental conversion and **\$0.10** per e‑mail, we simulate profit when mailing only the top *k%* ranked by predicted uplift. We report **uplift\@k**, mailed volume, incremental conversions, revenue gain, e‑mail cost, and **net profit**.

**Mens vs Control** (n ≈ 12,784 in this split)

| k    | uplift\@k | n\_mailed | incremental\_conv | revenue\_gain | email\_cost | net\_profit |
| ---- | --------: | --------: | ----------------: | ------------: | ----------: | ----------: |
| 5%   |  0.002617 |       639 |             1.672 |         25.08 |        63.9 |      −38.82 |
| 10%  |  0.008113 |     1,278 |            10.368 |        155.52 |       127.8 |      +27.72 |
| 20%  |  0.001479 |     2,556 |             3.781 |         56.72 |       255.6 |     −198.88 |
| 30%  |  0.003411 |     3,835 |            13.081 |        196.21 |       383.5 |     −187.29 |
| 100% |  0.006727 |    12,784 |            86.000 |      1,290.00 |     1,278.4 |      +11.60 |

**Womens vs Control** (n ≈ 12,808 in this split)

| k    | uplift\@k | n\_mailed | incremental\_conv | revenue\_gain | email\_cost | net\_profit |
| ---- | --------: | --------: | ----------------: | ------------: | ----------: | ----------: |
| 5%   |  0.006663 |       640 |             4.265 |         63.97 |        64.0 |       −0.03 |
| 10%  |  0.005087 |     1,280 |             6.511 |         97.67 |       128.0 |      −30.33 |
| 20%  |  0.002220 |     2,561 |             5.686 |         85.29 |       256.1 |     −170.81 |
| 30%  |  0.002495 |     3,842 |             9.586 |        143.78 |       384.2 |     −240.42 |
| 100% |  0.003096 |    12,808 |            39.648 |        594.72 |     1,280.8 |     −686.08 |

---

---

## Decisions & recommendations

1. **Mens E‑Mail:** Strong global lift. **Mail the top \~10%** by uplift for higher profit vs. blanket sends; keep a **holdout cell** to measure true incrementality.
2. **Womens E‑Mail:** Lift exists on average, but profitability is fragile. **Either pause** or **restrict to the top 5%** while we **improve the model** and validate economics.
3. **Economics validation:** Confirm **margin (\$15)** and **per‑send cost (\$0.10)** with Finance/CRM; small changes can flip ROI.
4. **Experimentation plan:**

   * Run a **multicell test** (Mens: top‑10% vs all vs control; Womens: top‑5% vs all vs control).
   * **Primary KPI:** incremental profit; **guardrails:** complaint rate, unsubscribe, deliverability.

---

---

## Modeling roadmap (next steps)

* **Feature enrichment:** customer tenure, seasonal flags, category affinity, historical spend frequency/recency, interaction terms; reduce leakage.
* **Algorithms:** X‑learner, DR‑learner, gradient boosting; tune trees/calibration; cross‑fitting.
* **Evaluation:** more robust **Qini/UPLIFT\@K** with cross‑validation; **calibration plots**.
* **Targeting policy search:** sweep *k* more finely (e.g., 5–25% in 1% steps) to maximize expected profit.

---

---

## Reproducibility

### Environment

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python main.py
```

### Project structure

* `data_loader.py` — loads Hillstrom data and prepares treatment/control splits.
* `eda.py` — randomization balance tables by segment.
* `ab_test.py` — conversion z‑tests; spend Welch + bootstrap CI.
* `uplift_model.py` — T‑learner (Random Forests + calibration), Qini curve/AUC, uplift\@k.
* `roi.py` — ROI simulation for top‑k targeting (margin & cost configurable).
* `main.py` — orchestrates EDA → A/B tests → uplift + ROI, printing the tables shown above.
* `requirements.txt` — Python dependencies.

**Expected console output** (abridged): randomization check (history/recency), three conversion z‑tests, two spend tests with CIs, **Qini AUC** for Mens/Womens vs Control, and **ROI tables** at k ∈ {5%, 10%, 20%, 30%, 100%}—matching the values in this README.

---

---

## Assumptions & caveats

* ROI uses fixed **margin = \$15** and **cost/email = \$0.10**; adjust to your business (consider contribution margin, returns, and variable costs).
* Uplift performance depends on **features, calibration, and sample**; negative Qini for Mens vs Control indicates **policy should be conservative** until the model improves.
* All statistics reported here derive from the **same data split** used in `main.py`; for production, use **out‑of‑time validation** and **separate test markets**.

---

---

## How to extend

* Add richer covariates and re‑run `main.py`.
* Replace RF with gradient boosting and re‑evaluate Qini/ROI.
* Add a notebook with **plots of Qini curves** and **profit vs. k** for stakeholder presentations.
