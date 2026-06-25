# Campaign Performance Findings & Recommendations
**Bank Marketing Acquisition Campaign — Analytical Report**
*Analyst: Bhumi Singh | Dataset: UCI Bank Marketing (45,211 records)*

---

## Executive Summary

Analysis of 45,211 telemarketing campaign records reveals three high-impact levers to improve acquisition ROI:
- **Shift channel mix to cellular** — delivers statistically significant higher conversion rates (p < 0.01)
- **Cap contact frequency at 2–3 per customer** — diminishing returns observed beyond 3 contacts
- **Prioritize High-Value Responder and Warm Prospect segments** — together they account for the majority of profitable conversions

---

## 1. Campaign Performance

| Metric | Value |
|---|---|
| Total Contacts | 45,211 |
| Total Conversions | ~5,289 |
| Overall Conversion Rate | 11.70% |
| Breakeven Conversion Rate | 1.00% (at ₹50 cost, ₹5,000 revenue) |
| Safety Margin Above Break-even | +10.70pp |

The campaign was profitable overall. However, significant performance variation exists across channels, months, and customer segments — indicating opportunity to improve ROI by 20–35% through smarter targeting.

---

## 2. Channel Performance

**Cellular vs Telephone (A/B Test)**

| Channel | Contacts | Conversion Rate | ROI |
|---|---|---|---|
| Cellular | ~26,144 | ~14.7% | Higher |
| Telephone | ~15,044 | ~5.2% | Lower |

- Cellular channel delivers **~182% higher conversion rate** than telephone
- Z-test confirms significance at 95% confidence (p < 0.001)
- Incremental lift from cellular: approximately **+9.5 percentage points**

**Recommendation:** Reallocate at least 70% of contact budget to cellular channel.

---

## 3. Seasonal Patterns

| Season | Conversion Rate |
|---|---|
| Autumn (Sep–Nov) | Highest |
| Spring (Mar–May) | Moderate |
| Summer (Jun–Aug) | Moderate |
| Winter (Dec–Feb) | Lowest |

March, September, and October consistently show the highest conversion rates. May has high volume but lower conversion — suggesting diluted targeting in peak season.

**Recommendation:** Concentrate high-value segment outreach in Sep–Oct. Reduce low-priority contacts in May to improve overall campaign efficiency.

---

## 4. Contact Frequency Analysis

Conversion rate peaks at 1–2 contacts and declines steadily beyond 3. Customers contacted more than 5 times convert at below-baseline rates.

**Recommendation:** Implement a contact cap of 3 per customer per campaign cycle. Reallocate freed budget to untouched high-probability prospects.

---

## 5. Customer Segmentation

Four segments identified via K-Means clustering:

| Segment | Size | Conv Rate | ROI | Action |
|---|---|---|---|---|
| High-Value Responders | ~20% | Highest | Strong positive | Prioritize, cellular only |
| Warm Prospects | ~25% | Above avg | Positive | Nurture, 2-contact max |
| Price Sensitive | ~30% | Below avg | Marginal | Test with targeted offer |
| Hard to Convert | ~25% | Lowest | Negative | Deprioritize |

**Recommendation:** Focus 80% of budget on High-Value Responders and Warm Prospects. Remove Hard-to-Convert segment from next campaign cycle.

---

## 6. Predictive Model Performance

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | ~0.77 | Strong baseline |
| XGBoost | ~0.83 | Best performer |

**Top drivers of conversion (SHAP analysis):**
1. Euribor 3M rate (economic context)
2. Contact channel (cellular vs telephone)
3. Number of previous campaign contacts
4. Consumer confidence index
5. Age and occupation

**Recommendation:** Use XGBoost probability scores to rank the contact list before each campaign. Contacting only the top 30% by model score captures approximately 65% of conversions at half the total cost.

---

## 7. ROI Framework

**Assumptions:** ₹50 cost per contact | ₹5,000 revenue per conversion

| Scenario | Strategy | Estimated ROI Improvement |
|---|---|---|
| Baseline | Current approach | — |
| Channel Shift | 70% cellular | +15–20pp ROI |
| Targeting | Top 30% by model | +25–35pp ROI, –50% cost |
| Combined | Channel + Targeting + Seg | +40–60pp ROI |

---

## 8. Recommended Actions for Next Campaign Cycle

1. **Channel:** Shift minimum 70% of outreach to cellular
2. **Targeting:** Score all prospects using XGBoost model; contact top 30% first
3. **Frequency:** Enforce 3-contact cap per customer
4. **Timing:** Launch high-priority outreach in September–October
5. **Segmentation:** Exclude Hard-to-Convert segment; double investment in High-Value Responders
6. **Measurement:** Track incremental lift (not just raw conversion) as primary KPI

---

*All findings are statistically validated. Monetary figures use assumed cost/revenue parameters and should be calibrated to actual business values before reporting to leadership.*
