import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from sklearn.model_selection import GridSearchCV
import gc
import itertools
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss
)

from constants import *

def train_model_bootstrap(df, model, n_bootstrap, test_size=0.2):
    # # Sepsis data
    # if df.shape[1] == 120:
    #     columns_to_drop = ['SepsisFlag', 't_0.05', 't_0.075', 't_0.1', 'HomogeneityAttack']
    # else:
    #     columns_to_drop = ['SepsisFlag', 't_0.05', 't_0.075', 't_0.1', 'HomogeneityAttack', 'cluster']

    # # Adult data
    # if df.shape[1] == 15:
    #     columns_to_drop = ['income_ >50K']
    # else:
    #     columns_to_drop = ['income_ >50K', 'cluster']

    # German credit data
    if df.shape[1] == 21:
        columns_to_drop = ['credit_risk_good']
    else:
        columns_to_drop = ['credit_risk_good', 'cluster']
    

    # Lists to store metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    losses = []
    cm_sum = np.zeros((2, 2))  # Assuming binary classification

    for i in range(n_bootstrap):
        # Prepare data
        X = df.drop(columns=columns_to_drop)
        # y = df["SepsisFlag"]
        # y = df["income_ >50K"]
        y = df["credit_risk_good"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        # Train model
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None  # Check if model supports predict_proba

        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        # Compute Binary Cross-Entropy
        losses.append(log_loss(y_test, y_prob))      

        if y_prob is not None:
            aucs.append(roc_auc_score(y_test, y_prob))
        
        # Sum confusion matrices for averaging
        cm = confusion_matrix(y_test, y_pred)
        cm_sum += cm

    # Compute average metrics
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1_score = np.mean(f1_scores)
    avg_auc = np.mean(aucs) if aucs else None  # Handle cases where AUC is not available
    avg_loss = np.mean(losses)
    avg_cm = cm_sum / n_bootstrap  # Averaged confusion matrix

    return avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_auc, avg_loss, avg_cm


