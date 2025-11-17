# 📧 Hillstrom Email A/B Test & Uplift Modeling  
*A/B testing • Uplift modeling • Incremental ROI • Qini curves*

---

## 🚀 Overview

This project analyzes the **Hillstrom 2008 Email Marketing Experiment** to answer:

**“Should we send marketing emails, and to whom, to maximize incremental profit?”**

We combine:

- Classical **A/B testing** (conversion lift, spend lift)  
- **Uplift modeling (T-learner)** to identify incremental impact  
- **Top-k% ROI simulation**  
- **Qini curves** to evaluate uplift performance  

All analysis can be reproduced with `main.py`.

---

## 📊 Key Findings

### **1. A/B Test Results**

**Mens Email vs Control**  
- Conversion: **1.253% vs 0.573%** (Δ = +0.681 pp, p < 1e-12)  
- Spend lift: **+$0.770** per customer  

**Womens Email vs Control**  
- Conversion lift: **+0.311 pp** (p = 0.00016)  
- Spend lift: **+$0.424**

**Conclusion:** Both campaigns generate statistically significant lift.

---

### **2. Uplift Modeling**

- **Mens:** Qini AUC = **–0.142**  
- **Womens:** Qini AUC = **+0.094**

**Interpretation:**  
- The women’s model shows weak but positive signal.  
- The men’s model does not generalize and underperforms random.

---

### **3. ROI Simulation (Mailing top-k%)**

**Mens Email**  
- Top 10% → **+$27.7** net profit  
- Mail all → **+$11.6**

**Womens Email**  
- Mostly negative ROI  
- Top 5% ≈ break-even

**Conclusion:**  
- Targeting **improves Mens ROI** significantly.  
- Women’s segment should be **paused or restricted**.

---

## 🧠 Recommendation

- Deploy **targeted uplift policy** for Mens (≈ top 10%).  
- Pause/improve Womens campaign.  
- Maintain an always-on control group to validate incrementality.

---

## 🛠 Project Structure

```
AB_Email_Testing/
│── main.py
│── requirements.txt
│── .gitignore
│
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── features/
│   │   └── eda.py
│   ├── models/
│   │   ├── ab_test.py
│   │   ├── uplift_model.py
│   │   └── roi.py
│   └── utils/
│       └── __init__.py
│
├── scripts/
│   ├── run_ab_tests.py
│   └── run_uplift.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_uplift_demo.ipynb
│
└── figures/
    ├── qini_mens_vs_no_email.png
    └── qini_womens_vs_no_email.png
```

---

## ▶️ How to Run

### **1. Setup**
```
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### **2. Run full analysis**
```
python main.py
```

### **3. Run individual components**

A/B test only:
```
python -m scripts.run_ab_tests
```

Uplift modeling only:
```
python -m scripts.run_uplift
```

---

## 📁 Data

Uses the publicly available **MineThatData / Hillstrom 2008 Email Dataset** containing:

```
recency, history, channel, mens, womens,
visit, conversion, spend, segment, newbie
```

Automatically loaded by `data_loader.py`.

---

## 🔧 Methods Used

- Two-proportion **z-test**  
- **Welch’s t-test** + bootstrap CI  
- **T-learner uplift model** with Random Forests  
- **Qini curve** & Qini AUC  
- **Top-k% incremental profit simulation**

---
