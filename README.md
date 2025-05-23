# Credit Card Fraud Detection Using Machine Learning
This project applies machine learning techniques to detect fraudulent credit card transactions and  then evaluates the machine learning models against performance metrics.
# Project Overview
Credit card fraud is an increasing problem, and detecting fraudulent transactions is crucial for financial security. This project explores fraud detection  using Logistic Regression, Naive Bayes, and Decision Trees. To address the challenge of class imbalance, common in fraud detection tasks, two resampling techniques were applied: SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic minority class samples, and RandomUnderSampler to reduce the size of the majority class The dataset used contains various transaction details such as distance from home, purchase amount, and usage of PIN or chip, along with a feature indicating whether the transaction was fraudulent.
After evaluation, the best performing model was selected and deployed as an interactive web application using Streamlit. The app allows users to input transaction details and receive a real-time prediction on whether the transaction is likely to be fraudulent. All necessary files for running the web interface are included in this repository
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
