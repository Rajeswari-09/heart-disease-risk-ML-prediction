# Heart Disease Risk Prediction using Machine Learning

A beginner-friendly machine learning project to predict the presence of heart disease using clinical health data.

This project was developed as a **student/fresher machine learning project** to learn and apply the complete ML workflow, from data analysis and preprocessing to model training, evaluation, explainability, and a simple Streamlit application.

> **Note:** This is an academic/portfolio project and is not intended for medical diagnosis.

---

## About the Project

The aim of this project is to build a machine learning model that predicts whether a person is likely to have heart disease based on selected clinical features.

The project uses the **UCI Heart Disease dataset** and compares different classification algorithms to understand which models perform better on the dataset.

### What I worked on

- Loaded and explored the dataset
- Checked missing values and data quality
- Performed exploratory data analysis (EDA)
- Prepared numerical and categorical features
- Converted the original target into a binary classification problem
- Trained multiple machine learning models
- Compared model performance
- Performed hyperparameter tuning
- Used SHAP and feature importance for basic model explainability
- Saved the trained model
- Created a simple Streamlit application for prediction

---

## Project Workflow

```text
Dataset
   ↓
Data Inspection
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Comparison
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Explainability
   ↓
Save Best Model
   ↓
Streamlit Application
```

---

## Dataset

The project uses the **UCI Heart Disease dataset**.

The dataset contains clinical information such as:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Maximum heart rate
- Exercise-induced angina
- ST depression
- Number of major vessels
- Thalassemia
- Other clinical measurements

The original target contains values from 0 to 4. For this project, it was converted into a binary target:

```text
0 → No heart disease
1 → Heart disease present
```

The dataset used in the project contains **920 records**.

---

## Technologies Used

### Programming
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Visualization
- Matplotlib
- Seaborn

### Explainability
- SHAP
- Permutation Importance
- Feature Importance

### Deployment
- Streamlit
- Joblib

### Development Tools
- Jupyter Notebook
- Git
- GitHub

---

## Machine Learning Models

I experimented with the following classification models:

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Support Vector Machine
6. Gradient Boosting
7. XGBoost

I used **5-fold Stratified Cross-Validation** to compare the models.

---

## Model Results

The main evaluation metrics used were:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Cross-Validation

The best cross-validation ROC-AUC was obtained by the **Support Vector Machine (SVM)** model.

| Model | CV ROC-AUC |
|---|---:|
| Support Vector Machine | **0.8858** |
| Logistic Regression | 0.8797 |
| Random Forest | 0.8724 |
| Gradient Boosting | 0.8629 |
| K-Nearest Neighbors | 0.8548 |
| XGBoost | 0.8509 |
| Decision Tree | 0.7085 |

### Test Set

After training and tuning the models, they were evaluated on a separate test set.

The **tuned XGBoost model** achieved the highest test ROC-AUC of **0.9108** among the evaluated models.

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Tuned XGBoost | 0.8207 | 0.8824 | 0.8451 | **0.9108** |
| Tuned Random Forest | 0.8207 | 0.8725 | 0.8436 | 0.9082 |
| Gradient Boosting | 0.8207 | **0.8922** | 0.8465 | 0.9057 |
| Tuned SVM | 0.8152 | 0.8824 | 0.8411 | 0.8975 |
| Tuned Logistic Regression | 0.7989 | 0.8431 | 0.8230 | 0.8918 |

These results are based on this dataset and should not be considered clinical performance.

---

## Preprocessing

The project uses a preprocessing pipeline for the input features.

### Numerical features

- Missing values → Median imputation
- Scaling → StandardScaler

### Categorical features

- Missing values → Most-frequent imputation
- Encoding → OneHotEncoder

Using a pipeline helped keep preprocessing consistent during model training and prediction.

---

## Hyperparameter Tuning

I used **GridSearchCV** to experiment with different parameter combinations for selected models.

Models tuned included:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost

The tuning process used ROC-AUC as the main scoring metric.

The tuned SVM had the best cross-validation ROC-AUC and was saved as the model used by the Streamlit application.

---

## Model Explainability

To understand the model better, I explored:

### Feature Importance

Used to identify important features in tree-based models.

### Permutation Importance

Used to see how model performance changes when individual features are shuffled.

### SHAP

Used to understand how different features contribute to individual model predictions.

Some of the features that showed importance in the analysis include:

- Chest pain type
- Exercise-induced angina
- ST depression
- Maximum heart rate
- Number of major vessels

These findings describe patterns learned by the model and do not represent medical or causal conclusions.

---

## Streamlit Application

A simple Streamlit application was created to demonstrate how the trained model can be used for prediction.

The application allows a user to enter clinical feature values and receive a model prediction.

### Run the application

```bash
streamlit run app/app.py
```

---

## Project Structure

```text
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
│
├── src/
│   ├── data_inspection.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── explain_model.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Rajeswari-09/heart-disease-risk-ML-prediction.git
```

### 2. Open the project folder

```bash
cd heart-disease-risk-ML-prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app/app.py
```

---

## What I Learned from This Project

Through this project, I practiced:

- Working with a real-world healthcare dataset
- Data cleaning and preprocessing
- Exploratory data analysis
- Classification algorithms
- Model comparison
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Basic explainable AI techniques
- Saving and loading ML models
- Building a simple ML application using Streamlit
- Organizing a machine learning project for GitHub

---

## Future Improvements

As a learning project, there are several areas I would like to improve:

- Try additional machine learning models
- Improve model calibration
- Experiment with different classification thresholds
- Add more explainability features
- Test the model on an external dataset
- Improve the Streamlit interface
- Add model monitoring and experiment tracking

---

## Disclaimer

This project is created for **learning, academic, and portfolio purposes**.

It is **not a medical diagnostic system** and should not be used to make medical decisions. The predictions are generated by a machine learning model trained on the available dataset.
