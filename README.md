# Survey Data Risk Stratification
### Predicting Mental Health Risk Levels Using Machine Learning

## Overview

This educational project analyses workplace mental health survey responses and compares machine learning models for classifying survey-derived risk levels.

The analysis explores factors such as family mental health history, treatment-seeking behaviour and interference with work. The labels represent a constructed risk score, rather than clinically verified mental health conditions.

## Objectives

- Prepare survey data for machine learning.
- Construct lower-risk and higher-risk labels using a defined scoring rule.
- Compare baseline classifiers with ensemble methods.
- Evaluate performance using macro-F1 and higher-risk recall.

## Target Construction

The project defines the following score:

```text
risk_score = (2 × work_interfere)
             + family_history
             + treatment
             + obs_consequence
```

The variables use their encoded values as defined in the notebook. A median-based split converts the score into two risk groups.

**Interpretation:** Models learn to predict these constructed labels. Their performance does not establish an ability to diagnose depression, anxiety or other mental health conditions.

## Models Compared

### Baseline Models

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbours (KNN)
- Naive Bayes
- Multilayer Perceptron (MLP)

### Ensemble Models

- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- XGBoost
- Voting, Stacking and Bagging classifiers

## Workflow

1. Explore survey responses and inspect data quality.
2. Prepare and encode features.
3. Construct the risk score and classification target.
4. Train and compare candidate models.
5. Tune hyperparameters using GridSearchCV and stratified cross-validation.
6. Evaluate classification performance and review errors.

## Evaluation

Two primary metrics guide model comparison:

- **Macro-F1:** Gives equal importance to each class when averaging F1 scores.
- **Higher-risk recall:** Measures how many observations labelled higher-risk the model correctly identifies.

Accuracy alone can obscure differences in performance between the two groups.

## Main Results

Ten classifiers were compared using 10-fold stratified cross-validation and tuned using GridSearchCV, with macro-F1 as the selection metric.

Tuned Extra Trees achieved the highest cross-validation macro-F1 of **0.6263**. On the held-out test set of **252 responses**, it achieved:

- **Accuracy:** 62.70%
- **Macro-F1:** 0.6236
- **Higher-risk recall:** approximately 59%
- **ROC-AUC:** 0.6955

The four variables used to construct the risk label were excluded from the predictors to prevent direct target leakage. Imputation, scaling and categorical encoding were performed within model pipelines.

These results measure prediction of a survey-derived label, not clinically validated mental health risk.

## Tools and Technologies

- Python
- Jupyter Notebook
- pandas and NumPy
- scikit-learn
- XGBoost

## Running the Project

1. Download or clone this repository.
2. Install the Python packages imported by the notebook.
3. Make the survey dataset available at the path specified in the data-loading cell.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run the cells in order.

Update any local file paths before running the notebook on another computer.

## Limitations

- Survey responses are self-reported and may contain reporting or sampling bias.
- The scoring rule is a project-defined proxy, not a validated clinical assessment.
- Treatment-seeking behaviour should not be interpreted as evidence of poor mental health on its own.
- If variables used to construct the target are also included as predictors, high performance may reflect reconstruction of the scoring rule. Feature selection should explicitly address this risk.
- Results may not generalise to other populations or workplaces.

The project is intended for learning and portfolio demonstration. It should not be used for clinical diagnosis, employment screening or decisions about individual employees.
