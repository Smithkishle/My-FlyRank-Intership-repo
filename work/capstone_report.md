# Capstone Report — CTR / Engagement Opportunity Scoring

- **Author:** FlyRank ML Research Intern
- **Lane:** CTR / Engagement Opportunity Scoring
- **Repo:** `https://github.com/Smithkishle/My-FlyRank-Intership-repo`
- **Date:** September 2026
- **Base Rate:** 10.6% (1,752 positives / 16,590 eligible pages)

---

## 1. Problem framing

Editorial and search marketing teams managing thousands of URLs cannot manually review every page each week. Traditional rank tracking identifies where a page ranks on Google, but fails to identify pages that rank favorably yet fail to capture their expected share of clicks.

- **Unit of Analysis:** One pseudonymized content item (page), identified by unique `content_id`.
- **Output:** A calibrated priority score (0–100), accompanied by human-readable reason codes (`severe_ctr_gap`, `page_one_prominence`, `stale_metadata`) and prescriptive action types (`review_title_meta`, `review_snippet_intent`, `improve_onpage_engagement`, `monitor`).
- **Human Action:** Editorial reviewers pull the top 50 ranked pages weekly to inspect and rewrite SERP title tags, meta descriptions, and snippet structured data.
- **Cost of a Wrong Call:** Reviewer hours are the scarce resource. Recommending a page that is not truly underperforming wastes human editorial time on pages with no upside. Conversely, missing a high-volume Page 1 page with a depressed CTR leaves thousands of organic visits on the table.
- **Why Machine Learning Helps:** Expected CTR falls off sharply and non-linearly with search position (e.g. 0.24% median for Page 1 vs 0.17% for Striking vs 0.09% for Page 3–5). Simple flat rules (e.g. "flag CTR < 1%") misclassify Page 2 pages as failures and miss Page 1 opportunities. Machine learning models capture non-linear interactions across historical traffic scale, position tiers, content formats, and query volume without fragile manual heuristics.

---

## 2. Data safety

- **Dataset Used:** FlyRank Anonymized Research Dataset (`data/raw/content_refresh_anonymized.csv`), comprising 30,000 rows across 32 pseudonymized clients over a trailing 90-day window.
- **Analysis Cohort:** 16,590 eligible pages with valid position data (`avg_position > 0`, excluding 1,205 "no-data" sentinels) and sufficient volume (`impressions_90d >= 500`, with activity in both 30-day sub-windows).
- **Excluded Columns (Leakage & Privacy Register):**
  - `trend_direction` & `trend_pct`: Excluded because they derive directly from the comparison windows and leak performance trends.
  - `clicks_last_30d`, `impressions_last_30d`, `sessions_last_30d`: Excluded from features because they belong to the target observation window.
  - `ctr`: Excluded because it aggregates both prior and recent windows, overlapping the target.
  - `content_id` & `client_id`: Excluded from model features; used exclusively for joins and client-holdout grouping.
  - `provider_used` & `model_used`: Excluded per data dictionary guidelines.
- **Privacy Confirmation:** No client names, domain names, URLs, page titles, or private search queries exist in the dataset or any generated output.

---

## 3. Baseline

- **Transparent Rule Formulation:**
  $$\text{baseline\_score} = (\text{expected\_ctr}_{\text{tier}} - \text{ctr}) \times \text{impressions\_90d} \times \text{staleness\_bonus}$$
  where $\text{expected\_ctr}_{\text{tier}}$ is the position tier's median CTR, and $\text{staleness\_bonus} = 1.25$ if $\text{days\_since\_last\_update} \ge 91$ (else $1.0$).
- **Fairness:** Computed on the exact same client-holdout test split as the machine learning models.
- **Baseline Test Performance:**
  - ROC AUC: **0.688**
  - Average Precision (PR AUC): **0.164** (vs base rate 0.102)
  - Precision@20: **0.050**
  - Precision@50: **0.120**
  - Precision@100: **0.170**

---

## 4. Model / analysis

- **Methods Evaluated:**
  1. *Logistic Regression:* Balanced class weights, StandardScaler.
  2. *Decision Tree:* Depth 5, min leaf 50, balanced class weights.
  3. *Random Forest:* 200 estimators, depth 10, min leaf 25, balanced subsampling.
  4. *Gradient Boosting:* 200 estimators, depth 4, learning rate 0.1, min leaf 25.
- **Target Formulation:** A time-aware forward label:
  $$\text{is\_ctr\_opportunity} = \mathbb{I}\left(\text{ctr}_{\text{last30}} < \text{quantile}_{25}(\text{ctr}_{\text{last30}} \mid \text{position\_tier})\right)$$
  Features use exclusively prior 30-day metrics (`*_prev_30d`) and static properties.
- **Clean Feature Set (28 features):**
  - *Numeric (20):* `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `log_impressions_prev30`, `log_clicks_prev30`, `ctr_prev30_safe`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`, `days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`.
  - *Categorical (8):* `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`.

---

## 5. Evaluation

- **Validation Design:** 80/20 `GroupShuffleSplit` on `client_id` (22 training clients / 15,348 pages; 6 test clients / 1,242 pages). Exactly zero client overlap.
- **Test Base Rate:** 10.23% (127 actual positives in test set).

### Model vs Baseline Comparison Table (Test Set)

| Model | ROC AUC | Avg Precision | Precision@20 | Precision@50 | Precision@100 | Lift over Baseline (P@50) |
|---|---:|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **0.974** | **0.813** | **1.000** | **0.980** | **0.750** | **8.2×** |
| Random Forest | 0.973 | 0.792 | 1.000 | 0.840 | 0.770 | 7.0× |
| Decision Tree | 0.972 | 0.729 | 0.750 | 0.860 | 0.730 | 7.2× |
| Logistic Regression | 0.964 | 0.764 | 0.950 | 0.900 | 0.740 | 7.5× |
| Baseline Heuristic Rule | 0.688 | 0.164 | 0.050 | 0.120 | 0.170 | 1.0× |

- **Error Analysis (Gradient Boosting @ 0.5 decision threshold):**
  - True Positives: 84 | False Positives: 34
  - True Negatives: 1,081 | False Negatives: 43
  - Low False Discovery Rate in top decile: In the top 50 picks, 49 of 50 are true opportunities (P@50 = 0.980).
  - Failure Mode Analysis: False positives occur on newly ranking pages where prior impression counts fluctuate; false negatives occur on pages with high scroll rates where SERP snippet competition suddenly increased.

---

## 6. Interpretation

- **What the Model Found:**
  - `log_clicks_90d` (0.4812 importance) and `position_tier` (0.2606 importance) dominate ranking. Pages with high historical impression volume but disproportionately low click accumulation represent the strongest signals.
  - Prior-window CTR (`ctr_prev30_safe`, 0.0227 importance) acts as a strong persistent indicator: pages under-capturing clicks in month 1 overwhelmingly continue under-capturing in month 2 without intervention.
- **Surprises and Negative Results:**
  - *Staleness alone is not predictive:* While the heuristic rule penalized stale pages, empirical audit showed 0–30d fresh pages and 91–180d stale pages had virtually identical opportunity rates (10.56% vs 10.58%).
  - *Word count has near-zero effect:* Word count between opportunity pages (median 2,881) and non-opportunity pages (median 2,990) did not differ meaningfully. Content length does not determine SERP click-through rate.

---

## 7. Recommendation

- **Operational Action Playbook:**
  - **Review Queue Breakdown (16,590 pages scored):**
    - `review_title_meta`: 796 pages (High confidence, severe CTR shortfall on Page 1)
    - `review_snippet`: 606 pages (Moderate gap, striking distance positions 4–10)
    - `review_engagement`: 6,778 pages (Low on-page engagement despite clicks)
    - `monitor`: 8,410 pages (Healthy or within expected tier boundaries)
- **Reviewer Workflow:**
  1. Pull weekly Top 50 from `review_title_meta`.
  2. Perform SERP intent check (confirm query is not zero-click / navigational).
  3. Optimize title tag to front-load high-intent modifiers; verify snippet is not truncated.
- **Limits:** Directional decision-support only. Does not prove causal refresh impact.

---

## 8. Reproducibility

- **Environment:** Python 3.12/3.14 with `pandas>=2.2`, `numpy>=1.26`, `scikit-learn>=1.4`, `matplotlib>=3.8`.
- **Random Seeds:** `random_state=42` across all splits, models, and sampling procedures.
- **Re-run Instructions:**
  ```bash
  git clone https://github.com/Smithkishle/My-FlyRank-Intership-repo.git
  cd My-FlyRank-Intership-repo
  pip install -r requirements.txt
  python scripts/run_all.py
  ```
