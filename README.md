# AdClickIQ – Advertisement Click-Through Rate Prediction

AdClickIQ is a machine learning project focused on **Click-Through Rate (CTR) prediction for online advertising**.

The objective of CTR prediction is to estimate the probability that a user will click on an advertisement based on available advertising and contextual features. Accurate CTR prediction is an important component of modern advertising and recommendation systems.

The project explores multiple approaches for tabular CTR prediction, including gradient-boosted decision trees, deep CTR models, feature engineering, negative sampling, cross-validation, and model ensembling.

> **Note:** This repository is based on an existing open-source CTR prediction implementation. The original implementation and attribution are retained in accordance with its MIT license.

---

## Problem Statement

For every advertisement impression, the system receives a set of numerical and categorical features.

The task is to predict:

```text
P(click | user, advertisement, context)
```

where:

* `1` represents a click
* `0` represents no click

The predicted probability can then be used by an advertising or recommendation system for ranking and decision-making.

---

## Key Features

* Click-Through Rate prediction for online advertising
* Numerical and categorical feature processing
* Label encoding for categorical variables
* Frequency/count-based feature engineering
* Negative sampling for large-scale datasets
* Stratified 5-fold cross-validation
* Gradient Boosting models
* Deep CTR models
* Model ensembling
* Sigmoid/log-odds based ensemble strategy
* Configurable training and prediction pipelines

---

## Models

### Gradient Boosting Models

#### LightGBM

LightGBM is used as one of the primary gradient-boosting models for tabular CTR prediction.

Configuration includes:

* Count-based features
* Stratified 5-fold cross-validation

#### CatBoost

CatBoost is also evaluated as a gradient-boosting approach.

Configuration includes:

* GPU training
* Count-based features
* Stratified 5-fold cross-validation

---

### Deep CTR Models

The project also explores neural-network-based CTR architectures.

#### xDeepFM

xDeepFM is used to model explicit and implicit feature interactions.

Configuration includes:

* Gauss Rank transformation
* Stratified 5-fold cross-validation

#### FiBiNET

FiBiNET is used to model feature importance and bilinear feature interactions.

Configuration includes:

* Gauss Rank transformation
* Stratified 5-fold cross-validation

---

## Feature Engineering

### 1. Label Encoding

Categorical features are transformed into numerical representations using label encoding.

This allows categorical information to be processed efficiently by the downstream models.

---

### 2. Count Features

Frequency information is calculated for categorical features and incorporated as additional model features.

For example:

```text
category → frequency of occurrence
```

This provides the model with information about how frequently a particular categorical value appears in the dataset.

---

### 3. Gauss Rank Transformation

Gauss Rank transformation is used for selected numerical features in the deep CTR models.

The transformation ranks numerical values and maps their distribution toward a Gaussian-like distribution.

---

## Negative Sampling

CTR datasets can contain very large numbers of negative examples.

Negative sampling is used to reduce the amount of data required during training while retaining representative negative examples.

The original experiments explored different sampling ratios and random seeds.

This approach can significantly reduce computational requirements when training on large-scale advertising data.

---

## Cross-Validation

The training pipeline uses **Stratified K-Fold cross-validation with 5 folds**.

The main objective is to maintain a similar distribution of positive and negative examples across the folds.

Conceptually:

```text
Dataset
   │
   ├── Fold 1 → Validation
   ├── Fold 2 → Validation
   ├── Fold 3 → Validation
   ├── Fold 4 → Validation
   └── Fold 5 → Validation
```

Each fold is used as the validation set while the remaining folds are used for training.

---

## Ensemble Learning

Multiple models can produce complementary predictions.

The project evaluates several ensemble strategies, including:

* Rank Ensemble
* Average Ensemble
* Weighted Average Ensemble
* Sigmoid Ensemble

### Sigmoid Ensemble

The sigmoid ensemble operates in log-odds space.

The sigmoid function is:

```text
σ(x) = 1 / (1 + e^(-x))
```

The inverse sigmoid, or logit function, is:

```text
σ⁻¹(x) = log(x / (1-x))
```

Predictions are transformed into log-odds, averaged, and then transformed back into probability space.

Conceptually:

```text
Model Predictions
       │
       ▼
    Logit
       │
       ▼
Average Log-Odds
       │
       ▼
   Sigmoid
       │
       ▼
Final CTR Prediction
```

The original experiments found the sigmoid ensemble to outperform the other evaluated ensemble strategies.

---

## Benchmark

The original competition experiments reported the following validation and leaderboard results:

| Model                       |     CV | Public LB | Private LB |
| --------------------------- | -----: | --------: | ---------: |
| LightGBM – 0.45 sampling    | 0.7850 |    0.7863 |     0.7866 |
| FiBiNET – 0.45 sampling     | 0.7833 |    0.7861 |     0.7862 |
| xDeepFM – 0.45 sampling     | 0.7819 |    0.7866 |     0.7867 |
| Wide & Deep – 0.45 sampling | 0.7807 |    0.7835 |     0.7837 |
| AutoInt – 0.45 sampling     | 0.7813 |    0.7846 |     0.7848 |
| CatBoost – 0.45 sampling    | 0.7765 |    0.7773 |     0.7778 |

### Ensemble Results

| Method                    |  Public LB | Private LB |
| ------------------------- | ---------: | ---------: |
| Rank Ensemble             |     0.7889 |          - |
| Average Ensemble          |     0.7892 |          - |
| Weighted Average Ensemble |     0.7891 |          - |
| Sigmoid Ensemble          | **0.7903** | **0.7905** |

These benchmark numbers belong to the **original competition implementation** and are not claimed as independent AdClickIQ results.

---

## Project Structure

```text
AdClickIQ/
│
├── config/
│   ├── sampling.yaml
│   ├── predict.yaml
│   ├── train.yaml
│   └── ensemble.yaml
│
├── input/
│
├── libs/
│
├── notebook/
│   └── eda.py
│
├── res/
│
├── scripts/
│   ├── covert_to_parquet.py
│   ├── sampling.py
│   ├── train.py
│   ├── predict.py
│   └── ensemble.py
│
├── environment.yaml
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Installation

The project uses Conda for environment management.

Create the environment using:

```bash
conda env create --file environment.yaml
```

Activate the environment:

```bash
conda activate <environment-name>
```

For PyTorch, install the appropriate version for your hardware and CUDA configuration according to the official PyTorch installation instructions.

---

## Running the Pipeline

### 1. Convert the dataset

```bash
python -m scripts.covert_to_parquet
```

### 2. Generate the sampled dataset

```bash
sh scripts/shell/sampling_dataset.sh
```

### 3. Run LightGBM experiments

```bash
sh scripts/shell/lgb_experiment.sh
```

### 4. Run CatBoost experiments

```bash
sh scripts/shell/cb_experiment.sh
```

### 5. Run xDeepFM experiments

```bash
sh scripts/shell/xdeepfm_experiment.sh
```

### 6. Run FiBiNET experiments

```bash
sh scripts/shell/fibinet_experiment.sh
```

### 7. Generate the ensemble prediction

```bash
python -m scripts.ensemble
```

---

## Example Training Configuration

A LightGBM experiment can be configured using:

```bash
MODEL_NAME="lightgbm"
SAMPLING=0.45

for seed in 517 1119
do
    python -m scripts.train \
        data.train=train_sample_${SAMPLING}_seed${seed} \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}

    python -m scripts.predict \
        models=${MODEL_NAME} \
        models.results=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed} \
        output.name=5fold-ctr-${MODEL_NAME}-${SAMPLING}-seed${seed}
done
```

---

## Experiments

The project explores several approaches and configurations.

### Approaches that showed useful improvements

* Count-based categorical features
* Negative sampling
* Stratified 5-fold cross-validation
* Gauss Rank transformation for selected deep CTR models
* Combining multiple model predictions

### Approaches that did not improve the original experiments

The original experiments reported limited or negative improvements from:

* Day-based cross-validation
* Day-specific features
* CatBoost with its `cat_features` parameter
* GPU-based XGBoost
* Hash features due to memory requirements
* DeepFM
* LightGBM DART

---

## Technologies

* Python
* Pandas
* NumPy
* LightGBM
* CatBoost
* PyTorch
* DeepCTR-PyTorch
* Hydra
* Scikit-learn
* PyArrow / Parquet

---

## Learning Outcomes

This project provides practical exposure to:

* Binary classification for advertising
* CTR prediction
* Tabular machine learning
* Categorical feature engineering
* Negative sampling
* Gradient boosting
* Deep CTR architectures
* Cross-validation
* Ensemble learning
* Model probability calibration
* Large-scale ML experimentation

---

## References

The project is based on techniques and ideas from the following work:

* [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://lightgbm.readthedocs.io/)
* [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
* [FiBiNET: Combining Feature Importance and Bilinear Feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433)
* [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
* [CatBoost](https://catboost.ai/)
* [Gauss Rank](https://github.com/ptts1993/gauss-rank)
* [Sigmoid Ensemble](https://www.kaggle.com/)

---

## Attribution

This repository is based on the open-source project:

**web-ctr-prediction**

Original repository:

https://github.com/ds-wook/web-ctr-prediction

The original project was published under the **MIT License**. The original `LICENSE` file is retained in this repository.

This repository reorganizes and presents the project under the **AdClickIQ** name for learning and experimentation purposes.
