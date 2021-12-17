#!/usr/bin/env python
# coding: utf-8

# --Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from clrprint import *
import graphviz 
from numpy import mean, std

# Algo Libraries
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# # CSCI 4380 Project

# Read datasets using Pandas
df_train = pd.read_csv('./data/poker-hand-training-true.data', header = None)
df_test = pd.read_csv('./data/poker-hand-testing.data', header = None)

# Rename columns
df_train.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
df_test.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']

# Extract features
X_train = df_train.loc[:,df_train.columns != 'Label']
X_test = df_test.loc[:,df_test.columns != 'Label']

# Extract labels
y_train = df_train['Label']
y_test = df_test['Label']
"""
# -- Exploratory Data Analysis
# Dataset Size
print("Train dataset: ")
print(df_train.head())
print(df_train.shape)
print()

print("Test dataset: ")
print(df_test.head())
print(df_test.shape)
print()

# Class distribution
print("Train dataset: ")
Y_train_dist = Y_train.groupby(Y_train).size()
print(Y_train_dist)
print()
print("Test dataset: ")
Y_test_dist = Y_test.groupby(Y_test).size()
print(Y_test)

# Class distribution: histogram
hist_df = pd.DataFrame()
hist_df['train'] = Y_train_dist/25010*100
hist_df['test'] = Y_test_dist/1000000*100
ax = hist_df.plot.bar(title="Class Distribution")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.ylim(0, 100)
plt.xlabel('Label')
plt.ylabel('Percentage of Dataset')
plt.xticks(rotation=0)
plt.show()
"""

# -- Pre-processing
def preprocess_data(data:pd.DataFrame):
    df = data.copy()
    df_copy = df[['C1', 'C2', 'C3', 'C4', 'C5']]
    df_copy.values.sort()
    df[['C1', 'C2', 'C3', 'C4', 'C5']] = df_copy
    df = df[['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']]
    return df
"""
clrprint("Training Dataset", clr='red')
print(df_train)
print()

clrprint("Training Dataset - Sorted", clr='red')
train_sorted = preprocess_data(df_train)
print(train_sorted)

clrprint("Testing Dataset", clr='red')
print(df_test)
print()

clrprint("Testing Dataset - Sorted", clr='red')
test_sorted = preprocess_data(df_test)
print(test_sorted)
"""

# Pre-processed data
train_sorted = preprocess_data(df_train)
test_sorted = preprocess_data(df_test)

# Pre-processed features
X_train_sorted = train_sorted.loc[:,train_sorted.columns != 'Label']
X_test_sorted = test_sorted.loc[:,test_sorted.columns != 'Label']
feature_names = list(X_train.columns)

# Pre-processed labels
y_train_sorted = train_sorted['Label']
y_test_sorted = test_sorted['Label']
#target_names = list(y_train.columns)

# -- Decision Tree
# Base Model
print("Decision Tree - Base Model")
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred, normalize=True)
print("Accuracy Score: ", accuracy)
print()
"""
data = export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True) 
graph = graphviz.Source(data) 
graph.render("Poker")
"""
# Pre-processed Data
print("Decision Tree - Pre-processed")
# Cross-Validation
kf = KFold(n_splits=10, random_state=1, shuffle=True)
clf = DecisionTreeClassifier()

scores = cross_val_score(clf, X_train_sorted, y_train_sorted, scoring='accuracy', cv=kf, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))





