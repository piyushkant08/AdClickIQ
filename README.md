# AdClickIQ

## ML-Based Advertisement Click-Through Rate Prediction

AdClickIQ is a machine learning system designed to predict the probability that a user will click on an advertisement based on numerical and categorical advertising features.

The project explores multiple machine learning approaches for large-scale CTR prediction, including gradient boosting, deep learning-based CTR architectures, feature engineering, cross-validation, and model ensembling.

---

## Problem Statement

In online advertising, every advertisement impression can be represented using a collection of user, advertisement, and contextual features.

The objective is to estimate:

```text
P(click | features)
```

where:

* `1` represents a clicked advertisement
* `0` represents an advertisement that was not clicked

The predicted probability can be used as a signal for advertisement ranking and optimization.

---

## Key Features

* Binary classification for advertisement click prediction
* Numerical and categorical feature processing
* Categorical feature encoding
* Frequency and count-based feature engineering
* Negative sampling for large datasets
* Stratified 5-fold cross-validation
* Gradient boosting models
* Deep CTR architectures
* Model ensembling
* Probability-based ensemble prediction
* Configurable training and prediction pipelines

---

## Machine Learning Models

### LightGBM

LightGBM is used for efficient gradient-boosted decision-tree based CTR prediction.

It is particularly suitable for tabular datasets containing numerical and categorical features.

Key techniques include:

* Count-based feature engineering
* Stratified 5-fold cross-validation
* Configurable sampling

---

### CatBoost

CatBoost provides another gradient boosting approach for evaluating CTR prediction performance.

The experiments include:

* Categorical feature processing
* Count-based features
* Stratified cross-validation
* GPU-supported training

---

### xDeepFM

xDeepFM is a deep CTR architecture designed to model both explicit and implicit feature interactions.

It is useful for learning complex relationships between categorical advertising features.

---

### FiBiNET

FiBiNET combines feature importance learning with bilinear feature interactions.

It allows the model to learn which feature interactions are more important for CTR prediction.

---

## Feature Engineering

### Categorical Features

Categorical features are transformed into numerical representations so that they can be consumed efficiently by the machine learning models.

### Count Features

Frequency-based features are generated from categorical variables.

For example:

```text
categorical value
       ↓
frequency in dataset
       ↓
additional numerical feature
```

These features provide additional information about the occurrence frequency of categorical values.

### Numerical Transformation

Selected numerical features can be transformed using rank-based normalization techniques for deep CTR models.

---

## Negative Sampling

CTR datasets can contain a very large number of negative impressions.

Negative sampling reduces the amount of training data while retaining representative negative examples.

Conceptually:

```text
Original Dataset
       │
       ├── Positive Clicks
       │
       └── Negative Impressions
                 │
                 ▼
          Negative Sampling
                 │
                 ▼
        Training Dataset
```

This reduces computational requirements during experimentation.

---

## Cross-Validation

The training pipeline uses **Stratified 5-Fold Cross-Validation**.

The dataset is divided into five folds while maintaining a similar distribution of positive and negative samples.

```text
             Dataset
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
    Fold 1    Fold 2    Fold 3
      │         │         │
      └─────────┼─────────┘
                │
          Fold 4 / Fold 5
```

Each fold is used as a validation set while the remaining folds are used for training.

---

## Ensemble Learning

Different models can capture different patterns in the advertising data.

AdClickIQ supports combining predictions from multiple models.

The evaluated ensemble approaches include:

* Rank Ensemble
* Average Ensemble
* Weighted Average Ensemble
* Sigmoid Ensemble

### Sigmoid Ensemble

The sigmoid ensemble combines model predictions in log-odds space.

The sigmoid function is:

```text
σ(x) = 1 / (1 + e^(-x))
```

The logit transformation is:

```text
logit(p) = log(p / (1-p))
```

The general process is:

```text
Model Predictions
       ↓
Log-Odds Transformation
       ↓
Combine Predictions
       ↓
Sigmoid Transformation
       ↓
Final CTR Probability
```

---

## Project Architecture

```text
                 Advertising Data
                        │
                        ▼
              Data Preprocessing
                        │
                        ▼
              Feature Engineering
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       LightGBM      CatBoost      Deep CTR
                                    Models
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                Model Predictions
                        │
                        ▼
                  Ensemble Layer
                        │
                        ▼
                 CTR Probability
```

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
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

Create the Conda environment using:

```bash
conda env create --file environment.yaml
```

Activate the environment:

```bash
conda activate <environment-name>
```

For GPU-based experiments, install the appropriate PyTorch and CUDA configuration for your system.

---

## Running the Project

### Convert Data

```bash
python -m scripts.covert_to_parquet
```

### Sampling

```bash
python -m scripts.sampling
```

### Training

```bash
python -m scripts.train
```

### Prediction

```bash
python -m scripts.predict
```

### Ensembling

```bash
python -m scripts.ensemble
```

Configuration files under `config/` can be modified to control sampling, training, prediction, and ensemble experiments.

---

## Evaluation

CTR prediction is evaluated as a probability prediction problem.

Important evaluation metrics include:

### ROC-AUC

Measures how effectively the model separates clicked and non-clicked advertisements.

### Log Loss

Measures the quality of predicted probabilities and penalizes confident incorrect predictions.

For CTR prediction, probability quality is particularly important because the model produces a click probability rather than only a binary label.

---

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* LightGBM
* CatBoost
* PyTorch
* DeepCTR-PyTorch
* Hydra
* PyArrow
* Parquet

---

## Learning Outcomes

Through this project, the following concepts are explored:

* Binary classification
* CTR prediction
* Feature engineering
* Categorical feature processing
* Gradient boosting
* Deep CTR models
* Feature interactions
* Negative sampling
* Cross-validation
* Ensemble learning
* Probability prediction
* Large-scale ML experimentation

---

## Attribution and License

This project incorporates an existing open-source CTR prediction implementation and is used for learning and experimentation.

The original MIT License is retained in this repository.

All original copyright and license notices remain applicable.

## References

* LightGBM
* CatBoost
* Wide & Deep Learning
* xDeepFM
* FiBiNET
* Deep CTR modeling
* Sigmoid-based ensemble methods
