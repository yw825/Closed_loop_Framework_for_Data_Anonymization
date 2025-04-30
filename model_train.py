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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss, hinge_loss
)
from sklearn.dummy import DummyClassifier

from constants import *

def train_model_bootstrap(df, name, model, n_bootstrap, test_size=0.2):
    # # Sepsis data
    # if df.shape[1] == 120:
    #     columns_to_drop = ['SepsisFlag', 't_0.05', 't_0.075', 't_0.1', 'HomogeneityAttack']
    # else:
    #     columns_to_drop = ['SepsisFlag', 't_0.05', 't_0.075', 't_0.1', 'HomogeneityAttack', 'cluster']
    # columns_to_drop = ['SepsisFlag', 'PatientIdentifier']

    # # Adult data
    # if df.shape[1] == 97:
    #     columns_to_drop = ['income_ >50K']
    # else:
    #     columns_to_drop = ['income_ >50K', 'cluster']

    # German credit data
    if df.shape[1] == 49:
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
    tps = []
    tns = []
    fps = []
    fns = []

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
        
        # Compute metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        tps.append(confusion_matrix(y_test, y_pred)[1, 1])
        tns.append(confusion_matrix(y_test, y_pred)[0, 0])
        fps.append(confusion_matrix(y_test, y_pred)[0, 1])
        fns.append(confusion_matrix(y_test, y_pred)[1, 0])

        # Compute loss values based on ML models
        if name == "SVM":
            y_score = model.decision_function(X_test)
            losses.append(hinge_loss(y_test, y_score))
        else:
            y_score = model.predict_proba(X_test)[:, 1]
            losses.append(log_loss(y_test, y_score))

        # Compute AUC using y_score (works for both cases)
        aucs.append(roc_auc_score(y_test, y_score))
        
    # # Compute average metrics
    # avg_accuracy = np.mean(accuracies)
    # avg_precision = np.mean(precisions)
    # avg_recall = np.mean(recalls)
    # avg_f1_score = np.mean(f1_scores)
    # avg_auc = np.mean(aucs) if aucs else None  # Handle cases where AUC is not available
    # avg_loss = np.mean(losses)
    # avg_cm = cm_sum / n_bootstrap  # Averaged confusion matrix

    return accuracies, precisions, recalls, f1_scores, aucs, losses, tps, tns, fps, fns


def get_model_standard(df, n_bootstrap, test_size=0.2):
    # # Sepsis data
    # if df.shape[1] == 120:
    #     columns_to_drop = ['SepsisFlag', 't_0.05', 't_0.075', 't_0.1', 'HomogeneityAttack']
    # else:
    #     columns_to_drop = ['SepsisFlag', 't_0.05', 't_0.075', 't_0.1', 'HomogeneityAttack', 'cluster']
    columns_to_drop = ['SepsisFlag', 'PatientIdentifier']

    # # Adult data
    # if df.shape[1] == 97:
    #     columns_to_drop = ['income_ >50K']
    # else:
    #     columns_to_drop = ['income_ >50K', 'cluster']

    # # German credit data
    # if df.shape[1] == 49:
    #     columns_to_drop = ['credit_risk_good']
    # else:
    #     columns_to_drop = ['credit_risk_good', 'cluster']
    
    # Prepare data
    X = df.drop(columns=columns_to_drop)
    y = df["SepsisFlag"]
    # y = df["income_ >50K"]
    # y = df["credit_risk_good"]

    baseline_loss = []
    for i in range(n_bootstrap):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)

        # Train model
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train)

        baseline_probs = dummy.predict_proba(X_test)
        baseline_loss.append(log_loss(y_test, baseline_probs))

    good_loss = np.mean(baseline_loss) * 0.9

    return good_loss

