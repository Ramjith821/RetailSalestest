# Retail Sales Prediction Using Machine Learning

## About the Project

This project applies machine learning to retail sales transactions to predict business outcomes and understand customer purchasing behaviour.

The project covers regression, classification, and clustering in **one Jupyter Notebook**, organised into Review 1 and Review 2 sections.

## Dataset

* **Dataset:** Retail Sales Transactions
* **File:** `Sales_transactions_2022_2025.csv`
* **Format:** CSV

The dataset contains information about customers, products, sales, quantities, discounts, profits, order dates, and returns.

## Machine Learning Algorithms

### 1. Regression Algorithms

Regression is used to predict a continuous numerical target, such as profit.

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor (SVR)
10. K-Nearest Neighbors Regressor

The regression algorithms implemented in the project should be listed here using their exact names from the notebook.

### 2. Classification Algorithms

Classification is used to predict whether a transaction will be returned using `Return_Flag` as the target variable.

1. Logistic Regression
2. K-Nearest Neighbors
3. Naive Bayes (Gaussian)
4. Decision Tree Classifier
5. Support Vector Machine (SVC)
6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. MLP (Multi-Layer Perceptron) Classifier

Classification performance is evaluated using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC where applicable. Since the return target is imbalanced, particular attention is given to identifying returned transactions.

### 3. Clustering Algorithms

Clustering is used to group customers according to their purchasing behaviour.

* K-Means Clustering
* Agglomerative Clustering

The analysis includes the Elbow curve, Dendrogram, PCA and t-SNE visualisations, and Silhouette, Davies–Bouldin, and Calinski–Harabasz scores.

## Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* Matplotlib and Seaborn
* Jupyter Notebook
* Streamlit (planned web application)

## How to Run the Project

1. Download or clone the project repository.

2. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit joblib
   ```

3. Place the CSV dataset in the appropriate directory.

4. Open the Jupyter Notebook.

5. Run the notebook cells sequentially.

## Streamlit Application

A Streamlit application is planned to integrate the three machine learning tasks:

* **Regression:** Predict the numerical target used in the notebook.
* **Classification:** Predict transaction return status.
* **Clustering:** Explore customer segments and purchasing behaviour.

The application will use the trained models and their corresponding preprocessing steps.

## Project Goal

The goal is to apply and compare machine learning algorithms on retail transaction data, predict relevant business outcomes, identify customer segments, and demonstrate the results through an interactive application.
