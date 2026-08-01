# Comparing Classifiers: Bank Marketing Term Deposit Prediction

## 1. Problem Statement

**Objective:** Analyze client and campaign data from a Portuguese bank's telephone marketing campaigns to understand which factors influence whether a client subscribes to a term deposit. Using exploratory data analysis (EDA) and four classification models — K-Nearest Neighbors, Logistic Regression, Decision Trees, and Support Vector Machines — we identify patterns across demographics, campaign contact behavior, and economic context to determine which clients are most likely to subscribe.

**Goal:**
- Understand patterns in client subscription behavior
- Analyze how demographic, campaign-contact, and macroeconomic variables impact subscription
- Compare multiple classification models on predictive performance and training cost
- Generate actionable insights for a more efficient, targeted telemarketing strategy

## 2. Source

UCI Machine Learning Repository — [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
Collected from a Portuguese bank's own contact-center phone campaigns (May 2008 – November 2010); described in Moro, Cortez & Rita (2014), *A Data-Driven Approach to Predict the Success of Bank Telemarketing.*

a. Raw Data: `data/bank-additional/bank-additional-full.csv` (41,188 rows), `data/bank-additional/bank-additional.csv` (10% sample, 4,119 rows)
b. Python File: `prompt_III_solution.ipynb`

## 3. Data Loading and Overview

**Tasks:**
- Import required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- Load dataset (`bank-additional-full.csv`, semicolon-delimited)
- Inspect structure of data

**Key Checks:**
- First few rows → `head()`
- Data types and nulls → `info()`
- Value counts on target → `value_counts()`

**Purpose:** To understand the dataset before performing cleaning and analysis.

## 4. Data Cleaning

The dataset was inspected for missing and problematic values before analysis. There are **no true `NaN` values** anywhere in the 21 columns, but six categorical columns (`default`, `education`, `housing`, `loan`, `job`, `marital`) encode missing information using the literal string `"unknown"` instead. Rather than dropping these rows — which would discard over 20% of the data, since `default` alone is ~21% `"unknown"` — we retained `"unknown"` as its own category, since it is informative on its own (clients who won't disclose default status behave differently than those who report "no"). The `pdays` column uses a sentinel value of **999** to mean "never previously contacted," which was noted but not treated as a literal numeric distance. The `duration` column was excluded from all predictive models, since per the dataset documentation it is not known before a call takes place and leaks the outcome almost perfectly. Overall, the dataset required no row deletion and was suitable for exploratory analysis and modeling after these adjustments.

**Steps:**

4.1. Handle Missing Values
Identify categorical columns with `"unknown"` values; retain `"unknown"` as its own category rather than imputing or dropping.

4.2. Excluded the `duration` column
Removed from all realistic/deployable models due to target leakage (call outcome is known by the time duration is recorded).

4.3. Validate Data
Confirmed zero true nulls via `info()`; confirmed target `y` is binary (`"yes"`/`"no"`) and encoded it to 1/0.

**Purpose:** Ensure data quality and consistency before proceeding to analysis and modeling.

## 5. Exploratory Data Analysis (EDA)

**5.1. Overall Subscription Rate**
Compute overall subscription rate and understand baseline performance.
*Insight Focus:* What proportion of contacted clients subscribe to a term deposit?

**5.2. Target Distribution**
Bar plot of the `y` column.
*What to analyze:* Class balance/imbalance in the dataset.
*Expected Insight:* Dataset is heavily skewed toward "no" (non-subscribers).

**5.3. Client Demographics vs. Subscription**
Bar plots of subscription rate by `job`, `education`, `marital`, `default`, `housing`, `loan`; histogram of `age` by outcome.

**5.4. Economic Context**
Correlation heatmap across numeric/economic indicators (`emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`, `age`, `campaign`, `pdays`, `previous`).

## 6. Bank-Client-Only Baseline Model

Using only the seven bank client attributes (age, job, marital, education, default, housing, loan), a Logistic Regression model was compared against a majority-class dummy baseline and against KNN, Decision Tree, and SVM (default settings). None of the four models meaningfully beat the 88.7% baseline on this feature set alone, indicating that demographics by themselves carry only weak signal about subscription.

**Key Takeaway**
Bank-client demographics alone are not sufficient to reliably predict subscription; SVM in particular is not a practical choice on this feature set given its training time (~25 seconds vs. ~0.02 seconds for Logistic Regression) for no accuracy gain.

## 7. Improved Model: Campaign Timing + Economic Context

A second, expanded model added campaign-contact features (`contact`, `month`, `day_of_week`, `campaign`, `pdays`, `previous`, `poutcome`) and the five socio-economic indicators, while still excluding `duration`. All four models were retuned via `GridSearchCV` (scored on ROC AUC, with `class_weight='balanced'`) to properly account for the 11.3% minority class rather than optimizing for raw accuracy.

Frequent, recent prior contact and favorable/uncertain macroeconomic conditions are strongly associated with higher subscription likelihood, showing much stronger behavioral and contextual alignment than demographics alone. Clients contacted via cellular phone, and those contacted in specific months (notably quarter-end months), show meaningfully higher acceptance. Combining contact-recency, channel, and economic-context features substantially improves targeting effectiveness over the demographics-only model.

**Key Takeaway**
Subscription is driven far more by *when and how* a client is contacted, and by the surrounding economic climate, than by who the client is — making behavioral/contextual targeting the most effective strategy.

## 8. Key Findings

**Overall subscription rate: 11.3%** (4,640 of 41,188 contacts), indicating a low-response, high-imbalance problem where raw accuracy is a misleading metric.

**Bank-Client-Only Model**
- Baseline (majority class) accuracy: 88.73%
- Logistic Regression / SVM (default): 88.73% (i.e., no improvement over baseline)
- KNN (default): 87.89% test accuracy (below baseline — early overfitting signal)
- Decision Tree (default): 86.57% test accuracy vs. 91.78% train accuracy (clear overfitting)
- **Impact: demographics alone do not beat the naive baseline**

Demographic factors show only weak, inconsistent relationships with subscription. Students, retirees, and clients with an undisclosed (`"unknown"`) default status stand out, but overall separability from demographics alone is close to chance.

**Expanded Model (Campaign + Economic Context, Tuned)**

| Model | Test Accuracy | Test ROC AUC |
|---|---|---|
| Logistic Regression (tuned) | 83.5% | 0.805 |
| Decision Tree (tuned) | 86.9% | 0.804 |
| KNN (tuned) | 90.1% | 0.780 |
| SVM (tuned, 10% sample) | 83.6% | 0.778 |

**Impact:** ROC AUC improves from ~chance-level (demographics-only models could not separate the classes better than random guessing) to **0.78–0.80** once campaign timing and economic context are added — a substantial, consistent gain across every model type.

**Top Predictive Drivers (Logistic Regression coefficients & Decision Tree importances)**
- `emp.var.rate`, `cons.price.idx`, `euribor3m`, `nr.employed` — macroeconomic indicators dominate
- `contact` channel (cellular vs. telephone)
- `month` of contact
- `pdays` (recency of prior contact) — more recent contact associated with higher subscription likelihood

## Statistical Interpretation

- Macroeconomic/contextual variables (`emp.var.rate`, `euribor3m`, `nr.employed`, `cons.price.idx`) show the strongest association with subscription across both the linear (Logistic Regression) and non-linear (Decision Tree) models, and are highly correlated with one another as they track the same 2008–2010 economic cycle.
- Contact-recency (`pdays`) and channel (`contact`) further influence subscription likelihood, independent of demographics.
- The relationship between prior-contact recency and subscription suggests a directional trend: more recent prior contact is associated with higher current subscription likelihood, supporting a "warm lead" follow-up effect.
- Class imbalance (11.3% positive rate) means accuracy alone understates true model quality differences; ROC AUC was used throughout the improved-model comparison for a fairer, imbalance-robust evaluation.

## Conclusion

Clients who are already "in motion" — recently contacted, reachable by cellular phone, contacted during historically favorable months, and being approached during periods of greater economic uncertainty — are significantly more likely to subscribe to a term deposit. For example, the demographics-only model could not separate subscribers from non-subscribers better than chance, while adding campaign-timing and economic-context features raised ROC AUC to 0.78–0.80 across every model tested.

Demographics play a secondary role: some patterns exist (students and retirees subscribe more, clients with undisclosed default status subscribe less), but they are far weaker and less consistent than contextual and behavioral factors.

On the other hand, clients contacted via landline, contacted in historically weaker months, or with no recent prior contact are much less likely to subscribe, all else equal.

Overall, the key takeaway is that **timing- and context-driven targeting is far more effective than demographic-only targeting** for this bank's telemarketing campaigns. By focusing on the right contact channel, timing, and follow-up cadence — informed by current economic conditions — the bank can significantly improve subscription rates while reducing the total number of calls placed.

## Recommendations

- Prioritize cellular contact over landline wherever possible
- Concentrate campaign pushes in historically higher-performing months (e.g., quarter-end months)
- Re-engage clients with more recent prior contact first (`pdays`) rather than cold-calling
- Monitor macroeconomic indicators and expect/plan for higher response rates during periods of greater economic uncertainty
- Use the model to **rank** clients by predicted subscription probability and call down the list, rather than applying a fixed yes/no cutoff
- Favor the tuned Decision Tree or KNN model for deployment given their strong balance of accuracy, ROC AUC, and (for the Decision Tree) interpretability/training speed

## Next Steps

- Deploy the tuned Decision Tree or Logistic Regression model in a shadow/pilot mode alongside the existing calling process before full rollout
- Tune the decision threshold (not just the model) to the bank's actual cost/benefit trade-off between a wasted call and a missed subscriber
- Collect additional client-level financial data (e.g., account balance, product holdings, tenure) beyond demographics to further improve predictive power
- Periodically retrain the model as economic conditions evolve, since the strongest predictors identified are macroeconomic and will drift over time
- Run A/B tests comparing model-prioritized call lists against the current general-calling approach to validate real-world lift

## Repository Structure

```
.
├── README.md
├── prompt_III_solution.ipynb        # Main analysis notebook (start here)
├── CRISP-DM-BANK.pdf                # Background paper on the dataset/methodology
└── data/
    └── bank-additional/
        ├── bank-additional-full.csv # Full dataset (41,188 rows)
        ├── bank-additional.csv      # 10% random sample (4,119 rows), used for SVM tuning
        └── bank-additional-names.txt# Original UCI attribute documentation

jupyter notebook prompt_III_solution.ipynb
```
