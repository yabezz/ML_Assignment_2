# Bank Marketing Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit.

The project implements five classification algorithms on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit application is provided to allow users to upload test data, select a trained model, and view its evaluation results. The application loads previously saved models and does not retrain them.

---

## 2. Dataset Description

### Dataset

**Bank Marketing Dataset**

**Source:** UCI Machine Learning Repository

The Bank Marketing dataset contains information collected during direct marketing campaigns conducted by a Portuguese banking institution. The classification task is to predict whether a customer subscribed to a term deposit.

The selected `bank-full.csv` dataset contains:

- 45,211 instances
- 16 input features
- 1 binary target variable
- Target variable: `y`
- Target values: `yes` and `no`

The dataset therefore satisfies the assignment requirement of at least 500 instances and at least 12 features.

### Input Features

The 16 input features used in this project are:

1. age
2. job
3. marital
4. education
5. default
6. balance
7. housing
8. loan
9. contact
10. day
11. month
12. duration
13. campaign
14. pdays
15. previous
16. poutcome

### Target

The target variable is:

`y`

where:

- `yes` = customer subscribed to a term deposit
- `no` = customer did not subscribe to a term deposit

---

## 3. Data Preprocessing

The dataset contains both numerical and categorical variables.

The preprocessing pipeline performs the following operations:

### Numerical Features

- Missing-value imputation using the median
- Standardization using StandardScaler

### Categorical Features

- Missing-value imputation using the most frequent value
- One-hot encoding
- Unknown categories are handled using `handle_unknown="ignore"`

A stratified train-test split was used with:

- Test size: 20%
- Random state: 42
- Stratification based on the target variable

The preprocessing steps are included within the saved model pipelines so that the same transformations are automatically applied during prediction.

---

## 4. Machine Learning Models

### 4.1 Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a binary outcome.

### 4.2 Decision Tree

Decision Tree Classifier recursively divides the feature space using decision rules to classify observations.

### 4.3 K-Nearest Neighbors

KNN classifies an observation based on the classes of its nearest neighboring observations.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes applies Bayes' theorem while assuming conditional independence between features and Gaussian distributions for continuous variables.

### 4.5 Support Vector Machine

Support Vector Machine attempts to find an effective decision boundary between the two classes.

---

## 5. Model Performance Comparison

The following results were obtained on the same test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.9017 | 0.8589 | 0.6248 | 0.3998 | 0.4876 | 0.4494 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.8932 | 0.9046 | 0.6503 | 0.1881 | 0.2918 | 0.3105 |

---

## 6. Model-wise Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 90.12% and the highest AUC of 0.9056 among the five models.

Its precision was 0.6445, while recall was 0.3478. The model therefore provided a relatively balanced performance but did not identify a large proportion of positive cases.

Overall, Logistic Regression performed strongly and was particularly effective according to AUC.

### Decision Tree

Decision Tree achieved the highest accuracy of 90.17%.

It also achieved the highest F1 Score of 0.4876 and the highest MCC of 0.4494 among the evaluated models.

Its recall of 0.3998 was higher than Logistic Regression and KNN, while maintaining reasonable precision.

Based on the combination of accuracy, F1 Score, and MCC, Decision Tree provided the strongest overall performance in this experiment.

### KNN

KNN achieved an accuracy of 89.62%.

Its AUC was 0.8277, which was lower than Logistic Regression, Decision Tree, and SVM.

The F1 Score was 0.4340 and MCC was 0.4001. KNN therefore provided reasonable classification performance but was not the strongest model for this dataset.

### Naive Bayes

Naive Bayes achieved an accuracy of 85.48%, which was the lowest accuracy among the five models.

However, it achieved the highest recall of 0.5198. This indicates that it identified a larger proportion of positive cases than the other models.

Its precision was only 0.4059, meaning that a relatively large number of predicted positive cases were incorrect.

Naive Bayes may therefore be useful when identifying positive cases is more important than maintaining high precision.

### Support Vector Machine

SVM achieved an accuracy of 89.32% and an AUC of 0.9046.

It achieved the highest precision of 0.6503 among the five models.

However, its recall was only 0.1881 and its F1 Score was 0.2918. This indicates that the model was conservative in predicting the positive class and missed a relatively large number of actual positive cases.

---

## 7. Overall Winner

Based on the results obtained in this experiment, the **Decision Tree Classifier** is selected as the overall winner.

Decision Tree achieved:

- Highest Accuracy: 0.9017
- Highest F1 Score: 0.4876
- Highest MCC: 0.4494

Logistic Regression achieved the highest AUC, while SVM achieved the highest precision and Naive Bayes achieved the highest recall.

However, Decision Tree provided the strongest overall balance across accuracy, F1 Score, and MCC.

---

## 8. Streamlit Application

The Streamlit application provides the following functionality:

- Upload test CSV data
- Select one of the five trained models
- Load the saved model from the `model/` directory
- Calculate Accuracy
- Calculate AUC
- Calculate Precision
- Calculate Recall
- Calculate F1 Score
- Calculate MCC
- Display a confusion matrix
- Display a classification report
- Display a comparison of all five models

The application does not retrain the models.

---

## 9. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_results.csv
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

---

## 10. GitHub Repository

GitHub Repository:

`[Add GitHub repository link here after creating the repository]`

---

## 11. Live Streamlit Application

Streamlit Community Cloud Application:

`[Add deployed Streamlit application link here after deployment]`

---

## 12. Execution Environment

The project was developed and tested in the BITS Virtual Lab environment.

The local Streamlit application was tested using the BITS Virtual Lab browser environment.

---

## 13. Conclusion

Five classification models were trained and evaluated on the Bank Marketing dataset using the same preprocessing and test data.

The Decision Tree classifier provided the best overall performance based on Accuracy, F1 Score, and MCC. Logistic Regression produced the highest AUC, SVM produced the highest precision, and Naive Bayes produced the highest recall.

The completed Streamlit application provides an interactive interface for evaluating the saved models using uploaded test data.
