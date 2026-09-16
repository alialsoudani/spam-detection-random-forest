# Spam Detection Using Random Forest

## Overview

This project uses machine learning to classify emails as spam or not spam using the UCI Spambase dataset.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Random Forest

## Machine Learning Pipeline

1. Load the UCI Spambase dataset
2. Select relevant features
3. Handle missing values
4. Remove duplicate rows
5. Create a SpamScore feature
6. Handle outliers using IQR
7. Normalize numerical features using MinMaxScaler
8. Balance the classes using oversampling
9. Split the data into training and testing sets
10. Train a Random Forest classifier
11. Evaluate the model

## Evaluation

The model is evaluated using:

- Accuracy
- Classification Report
- Confusion Matrix
- Feature Importance

## Project Structure

```text
spam-detection-random-forest/
├── spam_detection.py
├── spam_train_data.csv
├── spam_test_data.csv
└── README.md
