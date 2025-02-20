# Credit Card Fraud Detection Using Machine Learning
This project applies machine learning techniques to detect fraudulent credit card transactions and  then evaluates the machine learning models against performance metrics.
# Project Overview
Credit card fraud is an increasing problem, and detecting fraudulent transactions is crucial for financial security. This project explores fraud detection currently using logistic regression and will later incorporate other machine learning models. The dataset used contains various transaction details such as distance from home, purchase amount, and usage of PIN or chip, along with a feature indicating whether the transaction was fraudulent.
# Features of This Project
- Exploratory Data Analysis (EDA) – Histograms, boxplots, and data distributions to understand transaction behavior.
- Data Preprocessing – Checking for missing values, feature scaling, and renaming columns for clarity.
- Data Balancing – Undersampling and (upcoming) oversampling to handle dataset imbalance.
- Machine Learning Model – Logistic Regression implemented to classify fraudulent vs. non-fraudulent transactions.
- Evaluation Metrics – Confusion matrix, precision, recall, and F1-score to assess model performance.
# Dataset
- home_distance – Distance from home where the transaction occurred.
- last_transaction_distance – Distance from the previous transaction.
- price_ratio – Ratio of transaction price compared to the median purchase price.
- card_used, pin_used, online_order – Binary indicators for transaction method.
- fraud – Target variable (1 = Fraudulent, 0 = Non-Fraudulent).

Source: [https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download]
