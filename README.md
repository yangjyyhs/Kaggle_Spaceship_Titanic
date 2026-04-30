# Spaceship Titanic — Kaggle 建模專案

Kaggle 競賽 [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) 的完整建模流程，從 raw data → imputation → feature engineering → modeling → stacking ensemble → hard rules。

**最佳成績：CatBoost 0.80967 (≈ Top 10%)**

完整 step-by-step 教學文章：見 [`Spaceship_Titanic_完整建模心得.md`](./Spaceship_Titanic_完整建模心得.md) (或 Notion 版)。

---

## 專案檔案

| 檔案 | 說明 |
|---|---|
| `V3_SpcaceShipPrediction_Kaggle.ipynb` | 主要 notebook — 從 imputing 一路到 stacking |
| `train.csv` / `test.csv` / `sample_submission.csv` | Kaggle 原始資料 |
| `Spaceship_Titanic_完整建模心得.md` | 給新手的完整教學文章 (Medium / Blog 用) |
| `v3_*_submission.csv` | 各 step 的 4 個模型 submission |
| `optimized_*_submission.csv` | 優化階段的 submission |
| `final_stacking*_submission.csv` | Stacking ensemble 的 3 個版本 |

---

## Notebook 架構

```
1. Import Package
2. Load Data
3. Model Selection                  # tune_and_evaluate 函式
4. Impute Data                      # 11 條規則 + KNN(k=3)
   ├── Engineering features (Total_spend, Group_ID, Last_name, Deck/CabinNum/Side)
   ├── CryoSleep ↔ Spending          # Rules 1+2
   ├── Age ≤ 12 → No Spending        # Rule 3
   ├── Same Group → HomePlanet/Dest  # Rules 4+5
   ├── Adult+Active+ZeroSpend→TRAPP  # Rule 10
   ├── Cabin Deck → HomePlanet       # Rule 8
   ├── Same Last_name → HomePlanet   # Rule 9
   ├── Age ≤ 17 → VIP=False          # Rule 7
   ├── Earth → VIP=False             # Rule 5
   ├── Europa Young / Deck T → VIP=F # Rules 6+7
   ├── Same Group → Cabin            # Rule 6
   ├── Cabin Neighbor Interpolation  # Rule 11
   └── KNN k=3 (fit on train, transform on both)

5. Modeling Helpers                  # run_step, run_step_oof
6. Step 1 Baseline                   # 10 features
7. Step 2 + Expense                  # +Total_spend, Has_Expenses
8. Step 3 + Cabin Spatial            # +Deck, Side, Cabin_Sector
9. Step 4 + Group/Family             # +Group_Size, Surname_TE (LOO smoothed)

10. Advanced Optimization
    ├── Opt Step 1: Group Label Propagation + Hard Rules
    ├── Opt Step 2: SHAP Feature Pruning (drop VIP)
    ├── Opt Step 3: Optuna Bayesian Tuning
    └── Opt Step 4: Pseudo-Labeling

11. Stacking Ensemble + Hard Rules    # XGB+LGBM+CatBoost → LR
```

---

## Kaggle Test Set 表現排行

### CatBoost (主力模型)

| Rank | Step | Score |
|---|---|---|
| 🥇 | optimized_step1_group_propagation | **0.80967** |
| 🥈 | optimized_step2_shap_pruned | 0.80921 |
| 🥉 | v3_step3_cabin | 0.80780 |
| 4 | v3_step4_group | 0.80500 |
| 5 | optimized_step4_pseudo_label | 0.80476 |
| 6 | optimized_step3_optuna_tuned | 0.80383 |
| 7 | v3_step1_baseline | 0.79424 |
| 8 | v3_step2_expense | 0.79354 |

### XGBoost

| Rank | Step | Score |
|---|---|---|
| 🥇 | optimized_step2_shap_pruned | **0.80687** |
| 🥈 | optimized_step1_group_propagation | 0.80617 |
| 🥉 | v3_step4_group | 0.80243 |

### LightGBM

| Rank | Step | Score |
|---|---|---|
| 🥇 | optimized_step1_group_propagation | **0.80500** |
| 🥈 | optimized_step2_shap_pruned | 0.80430 |

### Random Forest

| Rank | Step | Score |
|---|---|---|
| 🥇 | optimized_step1_group_propagation | **0.79985** |
| 🥈 | v3_step4_group | 0.79658 |

### Stacking Ensemble

| 版本 | Score |
|---|---|
| `final_stacking_userrules` (CryoSleep+ZeroSpend rule) | **0.80570** ⭐ |
| `final_stacking_norules` (no hard rules) | 0.80406 |
| `final_stacking` (+ in-test-group consensus) | 0.74959 ❌ |

---

## 6 個關鍵教訓

1. **資料清理 ≠ 填平均值** — 用資料規則 (CryoSleep → 0 spending) 比 mean imputation 強很多
2. **驗證每一條規則** — 從別人 notebook 抄的規則 11 條裡有 2 條完全錯
3. **Target Encoding 必須 OOF + LOO** — 否則 in-sample leak 會讓模型在 train 學假關聯
4. **越複雜不代表越好** — 最佳 CatBoost 是 Optimized Step 1，沒 Optuna、沒 stacking
5. **Hard rules 要保守** — In-test-group consensus 看似合理但 drop 5.6%
6. **CV ≠ Kaggle** — Optuna 找 CV 最佳反而 overfit；要忍住過度優化 CV 的誘惑

---

## 環境需求

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost shap optuna missingno
```

Python 3.10+，notebook 推薦 Jupyter / VSCode。
