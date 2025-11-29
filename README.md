# American Express Offerings Personalization Challenge

This repository contains the solution for the **American Express Campus Challenge 2025** (Amex Offerings Personalization). The goal of this data science challenge was to predict the probability that a customer will click on a given Amex offering, given that they have already seen it.

Our final solution is a LightGBM model trained on 320 heavily-engineered features, which achieved a **MAP\@7 score of 0.623** on the holdout validation set. This represents a \~17% absolute performance gain over the baseline model.

## Key Resources

* **[Final Presentation](https://drive.google.com/file/d/1sOVyTmuxhXn0Fk8sWwgjadTRuIhmT2Ry/view?usp=sharing)**: A slide deck detailing our final solution, methodology, and key findings.
* **[Final Submission CSV](https://drive.google.com/file/d/1_O8QieK9x76Ap4NTYNc3wXGSR_47Ud7i/view?usp=sharing)**: The final submission file with predicted probabilities.
* **[Feature Importance Report](https://docs.google.com/spreadsheets/d/1CbMKPWqzkMuxSdTFwAh33ef7jD8QkRLq/edit?gid=1678302543#gid=1678302543)**: A CSV file ranking the 320 features used in the final model by their LightGBM importance.

## The Challenge

The problem statement required participants to build a model to rank-order Amex offers for customers. The key task was to predict the probability of a click given an impression, \$P(\text{Click} | \text{Impression})\$.

* **Evaluation Metric:** The competition was judged on **MAP\@7 (Mean Average Precision at 7)**. This metric measures the quality of the rank-ordering, assigning a higher weight to correctly predicted clicks at the top of the list.
* **Data:** The dataset included a main file with 366 masked features (covering customer profiles, offer details, and past interactions). We were also provided three supplemental datasets for feature engineering:

  1. **Offerings Data**
  2. **Events Data**
  3. **Transaction Data**

## Repository Structure

```
├── data/
│   ├── (Raw data files: train.parquet, test.parquet, add_trans.parquet, etc.)
│
├── code/
│   ├── 1_data_preparation.ipynb        # Handles data loading, merging, and time-based splitting
│   ├── 2_feature_engineering.ipynb     # Creates 420+ features from raw data
│   ├── 3_model_training.ipynb          # Trains final LightGBM model and runs Optuna tuning
│
├── reports/
│   ├── final_presentation.pdf          # (Link to be added)
│   ├── feature_importance.csv          # (Link to be added)
│
├── submission/
│   ├── final_submission.csv            # (Link to be added)
│
└── README.md                           # This file
```

## Methodology

Our solution is an end-to-end data science pipeline built on three core stages.

### 1. Data Preparation (Validation & Sampling)

This stage was critical for building a model that generalizes to real-world, time-series data.

* **Validation Strategy:** A traditional random split would cause data leakage. We implemented a **strict time-based validation split**. The data was sorted by the event timestamp (`id4`) and split into Training (80%) and Validation (20%) sets. This ensures we always validate on "unseen future data".
* **Handling Class Imbalance:** The dataset was highly imbalanced, with clicks (\$y=1\$) being much rarer than non-clicks (\$y=0\$). After experimenting, we **chose not to use explicit sampling** (e.g., SMOTE or random under-sampling). Instead, we relied on LightGBM's native ability to handle imbalance, which preserved the original data distribution and yielded better results.

### 2. Feature Engineering

This was the most impactful part of our solution, transforming raw signals into 420+ powerful predictive features.

* **Strategy:** We grouped features into three dimensions: User Dynamics, Offer Characteristics, and User-Offer Match.
* **Feature Selection:** We used LightGBM importance and Mutual Information to select the **top 320 features** for the final model.
* **Key Engineered Features:**

  * `time_since_last_interaction`: Our **#1 most important feature**. It captures the critical temporal context of user activity.
  * `offer_ctr_bucket`: Binned popularity score of the offer.
  * `log_ctr_f137`: Log-transformed user CTR (last 60 days).
  * `user_session_length`: Number of offers seen in the current user session.
  * `click_rate_30d`: User's click-per-impression rate over the last 30 days.
  * `days_since_offer_start`: The age of the offer when it was presented.
  * `engagement_score`: A combined measure of user activity and historical CTR.

### 3. Final Model Training

* **Model:** **LightGBM (Light Gradient Boosting Machine)**. It was chosen for its high speed, accuracy, and efficiency on massive datasets.
* **Hyperparameter Tuning:** We used **Optuna** for Bayesian Optimization to find the best hyperparameters.
* **Optimization Metric:** While the competition metric was MAP\@7, we tuned our model by targeting **PR-AUC (Precision-Recall Area Under the Curve)**. This metric is ideal for imbalanced classification and directly aligns with the business goal of ranking precision. Our final model achieved a **PR-AUC of 0.7529** on our holdout set.

## Results

Our iterative approach yielded significant performance gains at each step, validated on our time-based holdout set.

| Model Iteration                                    | MAP\@7 Score |
| :------------------------------------------------- | :----------- |
| Baseline (LGBM Standalone)                         | 0.456        |
| + Time-based Split + Preliminary Feature Engg.     | 0.562        |
| **Final Model (Time-split + Final Feature Engg.)** | **0.623**    |

Our final model achieved an **absolute gain of \~17%** over the baseline, demonstrating the power of robust validation and deep feature engineering.

## Team

* Aditya Kanagalekar
* Ashutosh Balasubramaniam
* Jahnavi Kumar

*IIT Guwahati*
