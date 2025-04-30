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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
from sklearn.model_selection import GridSearchCV
import gc
import itertools
from sklearn.utils import resample
import ast
import json
import re
import statsmodels.api as sm

from constants import *

def data_prep(df):

    # Convert columns with object dtype to category dtype
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')

    # df = df.set_index('PatientIdentifier')
   
    return df

def encode_categorical_from_file(df):
    
    # Apply one-hot encoding to categorical columns
    df_encoded = pd.get_dummies(df, drop_first=True)  # drop_first=True avoids dummy variable trap
    
    return df_encoded

def track_errors(X_test, y_test, y_pred):
    # Create new columns for Type I and Type II errors and initialize them to 0
    X_test['TypeI_Error'] = 0  # False Positive: model predicted 1, actual is 0
    X_test['TypeII_Error'] = 0  # False Negative: model predicted 0, actual is 1
    
    # Only update the rows corresponding to the test set
    # Type I error: model predicted 1 but true value is 0
    X_test.loc[X_test.index, 'TypeI_Error'] = ((y_pred == 1) & (y_test == 0)).astype(int)
    
    # Type II error: model predicted 0 but true value is 1
    X_test.loc[X_test.index, 'TypeII_Error'] = ((y_pred == 0) & (y_test == 1)).astype(int)
    
    return X_test



def calculate_nqi_cqi_stats(df, NQIs, CQIs):

    stats = {}

    # Compute mean for NQIs
    for nqi in NQIs:
        if nqi in df.columns:
            stats[nqi] = df[nqi].mean()
    
    # Compute mode for CQIs
    for cqi in CQIs:
        if cqi in df.columns:
            stats[cqi] = df[cqi].mode()[0]  # mode() returns a Series, take first mode

    return stats



def calculate_denominator(df, NQIs, CQIs):

    # For NQIs, use NumPy for vectorized calculation of squared deviations from the mean
    denominator_nqi = sum(np.sum((df[nqi].values - df[nqi].mean())**2) for nqi in NQIs if nqi in df.columns)
    
    # For CQIs, use NumPy to calculate how many different values exist compared to the mode
    denominator_cqi = 0
    for cqi in CQIs:
        if cqi in df.columns:
            mode_cqi = df[cqi].mode()[0]  # Get the most frequent category for the CQI
            # Use NumPy to count non-mode values
            denominator_cqi += np.sum(df[cqi].values != mode_cqi)
    
    # Total denominator is the sum of the variances for NQIs and deviations for CQIs
    denominator = denominator_nqi + denominator_cqi
    return denominator



def get_cqi_levels(df, CQIs):

    levels_dict = {}
    for cqi in CQIs:
        if cqi in df.columns:
            # Map each unique category in the CQI column to a unique index
            levels_dict[cqi] = {category: index for index, category in enumerate(df[cqi].unique())}
    
    return levels_dict




def get_nqi_bounds(df, NQIs):

    bounds = {}
    
    for nqi in NQIs:
        if nqi in df.columns:
            lower_bound = df[nqi].min()
            upper_bound = df[nqi].max()
            bounds[nqi] = {'lower_bound': lower_bound, 'upper_bound': upper_bound}
    
    return bounds



def calculate_information_loss(original_df, anonymized_df, NQIs, CQIs, denominator):

    # Sort original data and anonymized data by 'PatientIdentifier'
    original_df = original_df.sort_index()
    anonymized_df = anonymized_df.sort_index()
    
    # Compute squared differences for numerical quasi-identifiers
    num_loss = np.sum((original_df[NQIs].values - anonymized_df[NQIs].values) ** 2)

    # Compute categorical mismatch (0 if same, 1 if different)
    cat_loss = np.sum(original_df[CQIs].values != anonymized_df[CQIs].values)

    # Total information loss per record
    total_loss = num_loss + cat_loss

    infoloss=total_loss/denominator

    return infoloss


def normalize_data(value, min_value, max_value):

    return (value - min_value) / (max_value - min_value + 1e-6)



def clean_and_parse_cell(cell):
    # Replace NumPy array with just the numbers inside the brackets
    cell = re.sub(r'array\((\[.*?\])\)', r'\1', cell)

    # Replace np.float64(...) with float
    cell = re.sub(r'np\.float64\((.*?)\)', r'\1', cell)

    try:
        return ast.literal_eval(cell)
    except Exception as e:
        print(f"Could not parse cell: {e}")
        return {}


def convert_results_df(results_df):
    parsed_results = []

    for row_idx in range(results_df.shape[0]):
        row = []
        for col_idx in range(results_df.shape[1]):
            raw_cell = results_df.iloc[row_idx, col_idx]
            parsed_cell = clean_and_parse_cell(raw_cell)
            row.append(parsed_cell)
        parsed_results.append(row)

    return parsed_results


def plot_metric_trend_with_mean(results_df, metric_column, y_label, smooth_method='moving_avg', window_size=5, y_range=None):
    results = convert_results_df(results_df)  # Convert results

    plt.figure(figsize=(50, 20))  # Adjust figure size

    all_res = []  # Store all particle values per iteration

    for j in range(len(results[0])):  # Iterate over particles
        res = [np.mean(results[i][j][metric_column]) for i in range(len(results))]  # Extract metric values
        all_res.append(res)
        plt.plot(res, alpha=0.6, linewidth=0.5)  # Plot all particles

    # Convert to NumPy array for efficient calculations
    all_res = np.array(all_res)
    mean_trend = np.mean(all_res, axis=0)
    std_dev = np.std(all_res, axis=0)  # Compute standard deviation for shading

    # **Trend Smoothing**
    if smooth_method == 'moving_avg':
        smooth_trend = pd.Series(mean_trend).rolling(window=window_size, center=True).mean()
    elif smooth_method == 'lowess':
        smooth_trend = sm.nonparametric.lowess(mean_trend, np.arange(len(mean_trend)), frac=0.1)[:, 1]
    else:
        smooth_trend = mean_trend  # Default to mean if no smoothing is applied

    # **Plot Mean with Smoothed Trend**
    plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

    # **Shaded Region for Variability**
    #plt.fill_between(np.arange(len(mean_trend)), mean_trend - std_dev, mean_trend + std_dev, color='red', alpha=0.2)

    # **Apply Y-axis range if provided**
    if y_range is not None:
        plt.ylim(y_range[0], y_range[1])

    plt.ylabel(y_label, fontsize=50)  
    plt.xlabel('Iteration', fontsize=50)  
    plt.tick_params(axis='y', labelsize=50)
    plt.tick_params(axis='x', labelsize=50)
    plt.legend(fontsize=50, loc='upper right')

    plt.show()

def plot_global_best_trend(results_df, metric_column, y_label):
    results = convert_results_df(results_df)  # Convert results

    plt.figure(figsize=(50, 20))  # Adjust figure size

    best_res = [min([np.mean(results[i][j][metric_column]) for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values
    
    # smooth_trend = pd.Series(best_res).rolling(window=5, center=True).mean()

    plt.plot(best_res, linewidth=10)  # Plot all particles
    # plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

    plt.ylabel(y_label, fontsize=50)  
    plt.xlabel('Iteration', fontsize=50)  
    plt.tick_params(axis='y', labelsize=50)
    plt.tick_params(axis='x', labelsize=50)

    plt.show()

def plot_global_best_so_far(results_df, metric_column, y_label):
    results = convert_results_df(results_df)  # Convert results

    plt.figure(figsize=(50, 20))  # Adjust figure size

    best_res = [min([np.mean(results[i][j][metric_column]) for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values

    # Compute the global best found so far
    global_best_so_far = [best_res[0]]  # Initialize with the first value
    for i in range(1, len(best_res)):
        global_best_so_far.append(min(global_best_so_far[-1], best_res[i]))  # Update with minimum so far

    plt.plot(global_best_so_far, linewidth=10)  # Plot the global best trend

    plt.ylabel(y_label, fontsize=50)  
    plt.xlabel('Iteration', fontsize=50)  
    plt.tick_params(axis='y', labelsize=50)
    plt.tick_params(axis='x', labelsize=50)

    plt.show()