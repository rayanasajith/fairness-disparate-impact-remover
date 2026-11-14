# Bias Mitigation with the Disparate Impact Remover


## How to run

### Option 1: Google Colab (recommended)

1. Click the **Open in Colab** button above.
2. Select **Runtime → Run all**.

The first cell installs all dependencies. Another cell downloads the raw datasets into AIF360's data folder, so no manual setup is needed. A full run takes about 5–10 minutes.

### Option 2: Locally

```bash
pip install aif360 BlackBoxAuditing scikit-learn pandas numpy matplotlib tqdm jupyter
jupyter notebook disparate_impact_remover.ipynb
```

Then run all cells in order. Python 3.10 or 3.11 is recommended.

## Notebook structure

| Section | What it does |
|---|---|
| 0. Setup | Installs packages and imports libraries |
| 1. Data | Downloads the raw data and loads the three datasets |
| 2. Before mitigation | Computes DP_Diff, DIR and consistency on the original labels |
| 3. Train/test split | Makes a stratified 70/30 split and checks the stratification |
| 4. DI Remover + classifier | Defines the repair pipeline and tests it on Adult |
| 4b. Diagnostic | Compares AIF360's repair with the paper (synthetic data and quantiles) |
| 5. Full experiment | Runs all datasets × 5 seeds × 11 repair levels |
| 6. Before vs. after | Builds the comparison table and plots fairness/accuracy against the repair level |

## Datasets

| Dataset | Protected attribute | Privileged group | Favourable outcome | Features used |
|---|---|---|---|---|
| Adult | sex | Male | income > 50K | age, education-num, capital-gain, capital-loss, hours-per-week |
| German Credit | age | age ≥ 25 | good credit | month, credit_amount, investment_as_income_percentage, residence_since, number_of_credits, people_liable_for |
| COMPAS | race | Caucasian | no recidivism within 2 years | age, juv_fel_count, juv_misd_count, juv_other_count, priors_count |

Only numeric features are used, because the repair needs values that can be ranked. The paper made the same choice.

## Parameter settings

| Parameter | Value |
|---|---|
| Train/test split | 70/30, stratified on label × protected group |
| Random seeds | 0, 1, 2, 3, 4 (results reported as mean ± std) |
| Repair levels (λ) | 0.0, 0.1, …, 1.0 (λ = 0 is the unmitigated baseline, λ = 1 is full repair) |
| Classifier | `LogisticRegression(solver='liblinear', class_weight='balanced')` |
| Scaling | Min-max, fitted on the training set only |
| Protected attribute | Used by the repair, removed before training the classifier |
| Consistency | k = 5 nearest neighbours, standardised features, protected attribute excluded |

All settings are defined in the notebook cells and can be changed there, e.g. `SEEDS`, `TEST_SIZE`, `REPAIR_LEVELS` and `K_NEIGHBORS`.

## Metrics

- **DP_Diff (demographic parity difference):** P(favourable | unprivileged) − P(favourable | privileged). 0 means parity.
- **DIR (disparate impact ratio):** P(favourable | unprivileged) / P(favourable | privileged). 1 means parity; below 0.8 fails the 80% rule.
- **Consistency:** individual fairness (Zemel et al., 2013). 1 means similar people always get the same outcome.
- **Accuracy** and **balanced accuracy:** predictive performance.

## Main results (mean over 5 splits)

| Dataset | Setting | DP_Diff | DIR | Accuracy |
|---|---|---|---|---|
| Adult | Original labels | −0.195 | 0.360 | – |
| Adult | Baseline (λ = 0) | −0.138 | 0.658 | 0.752 |
| Adult | Full repair (λ = 1) | −0.048 | 0.870 | 0.745 |
| German | Original labels | −0.129 | 0.821 | – |
| German | Baseline (λ = 0) | 0.087 | 1.143 | 0.616 |
| German | Full repair (λ = 1) | 0.004 | 1.009 | 0.625 |
| COMPAS | Original labels | −0.097 | 0.840 | – |
| COMPAS | Baseline (λ = 0) | −0.204 | 0.698 | 0.682 |
| COMPAS | Full repair (λ = 1) | −0.104 | 0.835 | 0.681 |

## Output files

Running the notebook creates:

- `results_per_seed.csv`: all metrics for every dataset, seed and repair level
- `results_summary.csv`: the before/after comparison table
- `dir_accuracy_vs_repair_level.png`: DIR and accuracy against the repair level

## Note on the implementation

Section 4b of the notebook shows that AIF360's repair differs from the paper. With two groups, it maps one group onto the other group's distribution instead of a "median" distribution in between. On data with many tied values (such as Adult), even a full repair does not make the groups' distributions equal. See the report for details.

## Reference

M. Feldman, S. A. Friedler, J. Moeller, C. Scheidegger and S. Venkatasubramanian. *Certifying and removing disparate impact.* KDD 2015, pp. 259–268.
