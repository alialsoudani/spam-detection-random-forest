# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 12:29:51 2026

@author: bbtnh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================================
# 1- Load Dataset
# =========================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
df = pd.read_csv(url, header=None)

print("Initial Data:")
print(df.head())

# =========================================
# 2- Basic Data Understanding
# =========================================

print("\nDataset Info:")
print(df.info())

# =========================================
# 3- Select Important Features Only
# =========================================

selected_columns = [
    15,  # free
    23,  # money
    19,  # credit
    6,   # remove
    16,  # business
    17,  # email
    18,  # you
    20,  # your
    7,   # internet
    8,   # order
    51,  # !
    52,  # $
    54,  # capital average
    55,  # capital longest
    56,  # capital total
    57,  # spam label
]

df = df[selected_columns]

# =========================================
# Rename Columns
# =========================================

df.columns = [
    "Free",
    "Money",
    "Credit",
    "Remove",
    "Business",
    "Email",
    "You",
    "Your",
    "Internet",
    "Order",
    "ExclamationMark",
    "DollarSign",
    "CapitalAverage",
    "CapitalLongest",
    "CapitalTotal",
    "Spam",
]

print("\nSelected Features:")
print(df.head())

# =========================================
# 4- Handling Missing Values
# =========================================

df.iloc[0:20, 0] = np.nan
df.iloc[40:60, 3] = np.nan
df.iloc[80:100, 5] = np.nan

print("\nMissing Values per Column:")
print(df.isnull().sum())

df = df.fillna(df.median())

print("\nAfter handling missing values:")
print(df.isnull().sum())

# =========================================
# 5- Handling Duplicates
# =========================================

print("\nDuplicate rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# =========================================
# 6- Feature Engineering
# =========================================

df["SpamScore"] = (
    df["Free"]
    + df["Money"]
    + df["Credit"]
    + df["DollarSign"]
    + df["ExclamationMark"]
)

print("\nAfter feature engineering:")
print(df[["SpamScore"]].head())

# =========================================
# 7- Handling Outliers
# =========================================

Q1 = df["SpamScore"].quantile(0.25)
Q3 = df["SpamScore"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["SpamScore"] >= lower)
    & (df["SpamScore"] <= upper)
]

print("\nAfter removing outliers:")
print(df.shape)

# =========================================
# 8- Feature Scaling
# =========================================

scaler = MinMaxScaler()

numeric_cols = [
    "Free",
    "Money",
    "Credit",
    "Remove",
    "Business",
    "Email",
    "You",
    "Your",
    "Internet",
    "Order",
    "ExclamationMark",
    "DollarSign",
    "CapitalAverage",
    "CapitalLongest",
    "CapitalTotal",
    "SpamScore",
]

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nAfter normalization (0-1):")
print(df.head())

# =========================================
# 9- Check Class Distribution
# =========================================

print("\nClass distribution (Spam):")
print(df["Spam"].value_counts())

# =========================================
# Separate Classes
# =========================================

class_0 = df[df["Spam"] == 0]
class_1 = df[df["Spam"] == 1]

if len(class_0) > len(class_1):
    majority = class_0
    minority = class_1
else:
    majority = class_1
    minority = class_0

# =========================================
# Oversampling
# =========================================

minority_upsampled = minority.sample(
    len(majority),
    replace=True,
    random_state=42,
)

df_balanced = pd.concat(
    [majority, minority_upsampled]
)

df_balanced = df_balanced.sample(
    frac=1,
    random_state=42,
)

print("\nAfter balancing:")
print(df_balanced["Spam"].value_counts())

# =========================================
# 10- Final Dataset
# =========================================

print("\nFinal dataset ready for Machine Learning:")
print(df_balanced.info())

# =========================================
# Features and Target
# =========================================

X = df_balanced.drop(columns=["Spam"])
y = df_balanced["Spam"]

# =========================================
# Train-Test Split
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y,
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# =========================================
# Save Train/Test Data
# =========================================

train_data = pd.concat(
    [X_train, y_train],
    axis=1
)

test_data = pd.concat(
    [X_test, y_test],
    axis=1
)

train_data.to_csv(
    "spam_train_data.csv",
    index=False
)

test_data.to_csv(
    "spam_test_data.csv",
    index=False
)

print("\nDatasets saved successfully!")

# ==================================================
# 11- Random Forest Training (RFT)
# ==================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)

rf_model.fit(X_train, y_train)

# ==================================================
# 12- Predictions
# ==================================================

y_pred = rf_model.predict(X_test)

print("\nPredictions for first 10 test samples:")
for i, pred in enumerate(y_pred[:10]):
    print(f"Sample {i+1}: Predicted Class = {pred}")

# ==================================================
# 13- Model Evaluation
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("Random Forest Results")
print("===================================")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==================================================
# 14- Confusion Matrix
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Spam", "Spam"]
)

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# ==================================================
# 15- Feature Importance
# ==================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()