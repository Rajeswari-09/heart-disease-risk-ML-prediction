# Heart Disease Risk Prediction --- Machine Learning

An end-to-end machine learning project for predicting the presence of
heart disease from clinical measurements using the **UCI Heart Disease
dataset**.

The project covers the complete ML lifecycle: data inspection,
exploratory data analysis, leakage-safe preprocessing, model comparison,
cross-validation, hyperparameter tuning, test-set evaluation, model
explainability, model persistence, and a Streamlit prediction interface.

> **Important:** This is an educational and research prototype. It is
> **not a medical diagnostic system** and has not been clinically
> validated.

------------------------------------------------------------------------

## Project Overview

The goal is to build a reproducible binary classification pipeline that
estimates whether heart disease is present based on patient clinical
measurements.

The original UCI `num` target represents disease severity from 0--4. For
this project, it is converted into a binary target:

-   `0` → No heart disease
-   `1` → Heart disease present (`num > 0`)

The project emphasizes not only predictive performance, but also **data
leakage prevention, clinically relevant evaluation metrics, model
interpretability, and reproducible deployment**.

------------------------------------------------------------------------

## Key Highlights

-   Worked with **920 patient records** from the UCI Heart Disease
    dataset.
-   Performed structured **EDA and data-quality inspection**.
-   Handled missing values using pipeline-based imputation.
-   Used **Stratified 80/20 train-test splitting**.
-   Built a leakage-safe `scikit-learn` preprocessing pipeline using
    `ColumnTransformer`.
-   Compared **7 classification algorithms**.
-   Used **5-fold Stratified Cross-Validation**.
-   Applied **GridSearchCV** to selected models.
-   Evaluated models using Accuracy, Precision, Recall, F1-score, and
    ROC-AUC.
-   Generated confusion matrices and combined ROC curves.
-   Applied **feature importance, permutation importance, and SHAP** for
    explainability.
-   Saved the trained preprocessing + model pipeline with `joblib`.
-   Built an interactive **Streamlit** application for prediction.
-   Included explicit medical-use limitations and research disclaimers.

------------------------------------------------------------------------

## Machine Learning Workflow

``` text
UCI Heart Disease Dataset
          │
          ▼
   Data Inspection
          │
          ▼
   Data Cleaning & EDA
          │
          ▼
 Binary Target Creation
          │
          ▼
 Stratified Train/Test Split
          │
          ▼
 Leakage-Safe Preprocessing
 ┌─────────────────────────────┐
 │ Numerical                    │
 │ Median Imputation            │
 │ StandardScaler               │
 │                              │
 │ Categorical                  │
 │ Most-Frequent Imputation     │
 │ One-Hot Encoding             │
 └─────────────────────────────┘
          │
          ▼
   Model Training
          │
          ├── Logistic Regression
          ├── K-Nearest Neighbors
          ├── Decision Tree
          ├── Random Forest
          ├── Support Vector Machine
          ├── Gradient Boosting
          └── XGBoost
          │
          ▼
  5-Fold Cross-Validation
          │
          ▼
  Hyperparameter Tuning
       GridSearchCV
          │
          ▼
 Held-Out Test Evaluation
          │
          ▼
 Explainability
 ├── Feature Importance
 ├── Permutation Importance
 └── SHAP
          │
          ▼
 Saved ML Pipeline
          │
          ▼
 Streamlit Prediction App
```

------------------------------------------------------------------------

## Dataset

**Dataset:** UCI Heart Disease --- multi-centre dataset

-   **Records:** 920
-   **Prediction task:** Binary classification
-   **Target:** `num`
-   **Target transformation:** `num > 0` → disease present
-   **Clinical input features used:** 13

### Numerical Features

-   Age
-   Resting blood pressure (`trestbps`)
-   Cholesterol (`chol`)
-   Maximum heart rate achieved (`thalch`)
-   ST depression (`oldpeak`)
-   Number of major vessels (`ca`)

### Categorical Features

-   Sex
-   Chest pain type (`cp`)
-   Fasting blood sugar (`fbs`)
-   Resting ECG (`restecg`)
-   Exercise-induced angina (`exang`)
-   ST segment slope (`slope`)
-   Thalassemia result (`thal`)

The identifiers and dataset-centre column are excluded from modelling.
The raw target column is also removed after the binary target is
created.

------------------------------------------------------------------------

## Data Preparation & Leakage Prevention

The project deliberately keeps preprocessing inside the ML pipeline
rather than transforming the complete dataset before splitting.

### Numerical preprocessing

``` text
Missing values
      ↓
Median imputation
      ↓
StandardScaler
```

### Categorical preprocessing

``` text
Missing values
      ↓
Most-frequent imputation
      ↓
One-Hot Encoding
```

`OneHotEncoder(handle_unknown="ignore")` is used so that unseen
categories at inference time do not cause deployment failures.

The complete preprocessing + classifier pipeline is fitted only on
training data and then reused for test evaluation and Streamlit
inference.

### Additional data-cleaning decisions

-   Zero values in `chol` and `trestbps` are treated as missing before
    imputation.
-   High-missingness variables such as `ca` and `thal` are retained
    rather than dropping affected rows.
-   Stratification is used during train/test splitting to preserve the
    target-class distribution.

------------------------------------------------------------------------

## Models Compared

Seven classification algorithms were evaluated:

  Model                    Approach
  ------------------------ --------------------------------
  Logistic Regression      Linear baseline
  K-Nearest Neighbors      Distance-based classifier
  Decision Tree            Tree-based classifier
  Random Forest            Ensemble of decision trees
  Support Vector Machine   Margin-based classifier
  Gradient Boosting        Sequential boosting
  XGBoost                  Gradient-boosted tree ensemble

------------------------------------------------------------------------

## Cross-Validation Results

Five-fold Stratified Cross-Validation was performed on the training set.

  ------------------------------------------------------------------------
  Model            CV ROC-AUC    CV Accuracy          CV F1      CV Recall
  ------------ -------------- -------------- -------------- --------------
  Support          **0.8858 ±         0.8139         0.8391         0.8769
  Vector             0.0212**                               
  Machine                                                   

  Logistic           0.8797 ±         0.8071         0.8286         0.8449
  Regression           0.0136                               

  Random             0.8724 ±         0.8085         0.8300         0.8426
  Forest               0.0283                               

  Gradient           0.8629 ±         0.7921         0.8174         0.8401
  Boosting             0.0203                               

  K-Nearest          0.8548 ±         0.8084         0.8329         0.8599
  Neighbors            0.0359                               

  XGBoost            0.8509 ±         0.7908         0.8156         0.8328
                       0.0411                               

  Decision           0.7085 ±         0.7133         0.7445         0.7518
  Tree                 0.0446                               
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## Hyperparameter Tuning

`GridSearchCV` was applied to:

-   Logistic Regression
-   Random Forest
-   XGBoost
-   Support Vector Machine

The tuning objective was **ROC-AUC**, using 5-fold Stratified
Cross-Validation on the training data only.

  --------------------------------------------------------------------------
  Model                             Tuned CV ROC-AUC Best Configuration
  --------------------- ---------------------------- -----------------------
  Support Vector                          **0.8858** `C=1`, `kernel='rbf'`
  Machine                                            

  Logistic Regression                         0.8819 `C=0.1`,
                                                     `solver='liblinear'`

  Random Forest                               0.8786 `n_estimators=100`,
                                                     `max_depth=None`,
                                                     `min_samples_split=5`

  XGBoost                                     0.8719 `n_estimators=100`,
                                                     `max_depth=3`,
                                                     `learning_rate=0.05`
  --------------------------------------------------------------------------

The final saved deployment model is the **tuned Support Vector
Machine**, selected using the highest training-only tuned CV ROC-AUC.

------------------------------------------------------------------------

## Held-Out Test Performance

The models were finally evaluated on a held-out test set that was not
used for hyperparameter tuning.

  -----------------------------------------------------------------------------
  Model            Accuracy    Precision       Recall           F1      ROC-AUC
  ------------ ------------ ------------ ------------ ------------ ------------
  XGBoost            0.8207       0.8108       0.8824       0.8451   **0.9108**
  (tuned)                                                          

  Random             0.8207       0.8165       0.8725       0.8436       0.9082
  Forest                                                           
  (tuned)                                                          

  Gradient           0.8207       0.8053   **0.8922**   **0.8465**       0.9057
  Boosting                                                         

  Random             0.8152       0.8148       0.8627       0.8381       0.9032
  Forest                                                           

  SVM (tuned)        0.8152       0.8036       0.8824       0.8411       0.8975

  SVM                0.8152       0.8036       0.8824       0.8411       0.8975

  Logistic           0.7989       0.8037       0.8431       0.8230       0.8918
  Regression                                                       
  (tuned)                                                          

  Logistic           0.8098       0.8190       0.8431       0.8309       0.8895
  Regression                                                       

  XGBoost            0.8261       0.8241       0.8725       0.8476       0.8754

  K-Nearest          0.7935       0.8019       0.8333       0.8173       0.8672
  Neighbors                                                        

  Decision           0.7120       0.7426       0.7353       0.7389       0.7091
  Tree                                                             
  -----------------------------------------------------------------------------

### Important interpretation

There are two different selection perspectives in this project:

-   **Deployment model:** tuned SVM, selected using training-only
    cross-validation ROC-AUC (`0.8858`).
-   **Best held-out test ROC-AUC:** tuned XGBoost (`0.9108`).
-   **Highest held-out recall:** Gradient Boosting (`0.8922`).

This distinction is intentional: the test set is used for final
comparison and reporting, not for selecting the deployed model.

For a medical screening context, recall is particularly important
because false negatives represent disease cases that the model fails to
identify. However, these results should **not** be interpreted as
evidence of clinical suitability.

------------------------------------------------------------------------

## Model Explainability

The project includes three complementary approaches to understand model
behaviour.

### 1. Tree-Based Feature Importance

Built-in feature importance is generated for compatible tree-based
models.

### 2. Permutation Importance

Permutation importance measures how model performance changes when
individual features are shuffled.

### 3. SHAP

SHAP is used to understand how individual features influence model
predictions.

The analysis identified variables such as:

-   Chest pain type (`cp`)
-   Exercise-induced angina (`exang`)
-   ST depression (`oldpeak`)
-   Maximum heart rate (`thalch`)
-   Number of major vessels (`ca`)

as important predictive signals in the trained model.

> Feature importance and SHAP values represent statistical associations
> learned by the model. They do **not** establish causal relationships
> between these variables and heart disease.

------------------------------------------------------------------------

## Streamlit Application

The project includes an interactive Streamlit application that loads the
saved ML pipeline and accepts patient features through a user-friendly
interface.

### Application capabilities

-   Clinical feature input form
-   Saved model loading
-   Binary prediction
-   Estimated model probability
-   No-disease / disease probability display
-   Clear research-use disclaimer

The app uses the same saved preprocessing + model pipeline used during
training, helping maintain consistency between training and inference.

### Run the application

``` bash
streamlit run app/app.py
```

------------------------------------------------------------------------

## Project Structure

``` text
heart-disease-risk-ML-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── heart_disease_uci.csv
│
├── models/
│   └── heart_disease_model.pkl
│
├── notebooks/
│   └── heart_disease_analysis.ipynb
│
├── reports/
│   ├── cv_results.csv
│   ├── model_results.csv
│   ├── tuning_results.csv
│   └── figures/
│       ├── 01_target_distribution.png
│       ├── 02_age_distribution.png
│       ├── 03_cholesterol_distribution.png
│       ├── 04_bp_distribution.png
│       ├── 05_numerical_distributions.png
│       ├── 06_correlation_matrix.png
│       ├── 07_categorical_vs_target.png
│       ├── 08_numerical_vs_target.png
│       ├── 09_key_relationships.png
│       ├── cv_roc_auc.png
│       ├── model_comparison.png
│       ├── roc_curves_all_models.png
│       ├── feature_importance_tree.png
│       ├── permutation_importance.png
│       ├── shap_summary.png
│       └── confusion matrices
│
├── src/
│   ├── data_inspection.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── explain_model.py
│
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## Tech Stack

**Programming & Data**

-   Python
-   Pandas
-   NumPy

**Machine Learning**

-   Scikit-learn
-   XGBoost

**Preprocessing & Evaluation**

-   ColumnTransformer
-   Pipeline
-   SimpleImputer
-   StandardScaler
-   OneHotEncoder
-   Stratified K-Fold Cross-Validation
-   GridSearchCV
-   ROC-AUC
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

**Explainable AI**

-   SHAP
-   Permutation Importance
-   Feature Importance

**Visualization**

-   Matplotlib
-   Seaborn

**Deployment**

-   Streamlit
-   Joblib

**Development**

-   Jupyter Notebook
-   Git / GitHub

------------------------------------------------------------------------

## Installation

Clone the repository:

``` bash
git clone https://github.com/Rajeswari-09/heart-disease-risk-ML-prediction.git
cd heart-disease-risk-ML-prediction
```

Create and activate a virtual environment:

### Windows

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

``` bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Running the Project

### Run the complete analysis notebook

``` bash
jupyter notebook notebooks/heart_disease_analysis.ipynb
```

### Run data inspection

``` bash
python src/data_inspection.py
```

### Run model training

``` bash
python src/train_models.py
```

### Run model evaluation

``` bash
python src/evaluate_models.py
```

### Run explainability analysis

``` bash
python src/explain_model.py
```

### Launch the Streamlit application

``` bash
streamlit run app/app.py
```

------------------------------------------------------------------------

## Key Engineering Practices Demonstrated

### Leakage-safe ML pipeline

Preprocessing and model training are combined in a single `scikit-learn`
pipeline so that imputers, scalers, and encoders learn only from
training data.

### Reproducible evaluation

The project uses fixed random seeds and Stratified K-Fold
Cross-Validation to make model comparison more consistent.

### Training-only model selection

Hyperparameter tuning uses the training data and cross-validation. The
held-out test set is reserved for final evaluation.

### Model persistence

The complete fitted pipeline is saved with `joblib`, allowing the
Streamlit application to reproduce the same preprocessing and prediction
workflow.

### Explainability

The project does not stop at predictive performance; it also
investigates which features influence model behaviour using multiple
interpretability techniques.

------------------------------------------------------------------------

## Limitations

This project should be interpreted as an ML research/portfolio prototype
rather than a clinical system.

1.  The dataset contains only 920 records, which is limited for clinical
    machine learning.
2.  Some variables have substantial missingness, particularly `ca` and
    `thal`.
3.  Imputation can introduce uncertainty.
4.  The dataset combines multiple study centres, which may introduce
    distribution differences.
5.  There is no external validation on an independent clinical cohort.
6.  Converting disease severity from 0--4 into a binary target loses
    severity information.
7.  The model has not undergone clinical validation or regulatory
    evaluation.
8.  Test-set performance should not be interpreted as real-world
    clinical performance.

------------------------------------------------------------------------

## Future Improvements

Potential next steps include:

-   External validation using an independent cohort
-   Probability calibration
-   Threshold optimization based on screening objectives
-   Cost-sensitive learning for false-negative reduction
-   Larger and more diverse datasets
-   LIME and counterfactual explanations
-   Model monitoring and data-drift detection
-   Experiment tracking and model versioning
-   Containerized deployment
-   Automated CI/CD for the ML application

------------------------------------------------------------------------

## Portfolio Takeaway

This project demonstrates an end-to-end **Data Science / Machine
Learning workflow**, moving from raw clinical data to an evaluated and
explainable model and finally to an interactive application.

It demonstrates practical experience with:

**Data Cleaning → EDA → Feature Engineering → ML Pipelines →
Cross-Validation → Hyperparameter Tuning → Model Evaluation →
Explainable AI → Model Deployment**

------------------------------------------------------------------------

## Disclaimer

This project is intended **only for educational and research purposes**.

It is **not a medical diagnostic tool**, has not been clinically
validated, and should not be used to make medical decisions. Model
predictions and probabilities reflect the behaviour of a machine
learning model trained on the available dataset and should not be
interpreted as medical advice.
