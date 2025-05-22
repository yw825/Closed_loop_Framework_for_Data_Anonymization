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
    # Remove newlines and extra whitespace
    cell = cell.replace('\n', ' ').replace('\r', ' ').strip()

    # Remove 'dtype=object' or similar dtype annotations
    cell = re.sub(r',?\s*dtype=[^\)]*', '', cell)

    # Replace NumPy array(...) with the content inside
    cell = re.sub(r'array\((\[.*?\])\)', r'\1', cell, flags=re.DOTALL)

    # Replace np.float64(...) with just the number
    cell = re.sub(r'np\.float64\((.*?)\)', r'\1', cell)

    try:
        return ast.literal_eval(cell)
    except Exception as e:
        print(f"Could not parse cell: {e}\nOriginal cell: {cell}")
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

def get_information_to_plot(results_df, metric_column, agg_func='max'):
    results = utils.convert_results_df(results_df)
    all_particle_metrics = []

    for j in range(len(results[0])):  # Iterate over particles
        if agg_func == 'mean':
            particle_metrics = [np.mean(results[i][j][metric_column]) for i in range(len(results))]
        elif agg_func == 'max':
            particle_metrics = [np.max(results[i][j][metric_column]) for i in range(len(results))]
        else:
            particle_metrics = [results[i][j][metric_column] for i in range(len(results))]

        all_particle_metrics.append(particle_metrics)

    return np.array(all_particle_metrics)  # shape: (num_particles, num_iterations)

def plot_metric_trend_for_each_particle(results_df, metric_column, y_label, 
                                        agg_func='max', smooth_method='moving_avg', 
                                        window_size=5, y_range=None):
    # Get the metric data from results
    all_particle_metrics = get_information_to_plot(results_df, metric_column, agg_func=agg_func)

    plt.figure(figsize=(50, 20))

    # Plot each particle
    for res in all_particle_metrics:
        plt.plot(res, alpha=0.6, linewidth=0.5)

    # Compute average and standard deviation
    mean_trend = np.mean(all_particle_metrics, axis=0)
    std_dev = np.std(all_particle_metrics, axis=0)

    # Apply smoothing
    if smooth_method == 'moving_avg':
        smooth_trend = pd.Series(mean_trend).rolling(window=window_size, center=True).mean()
    elif smooth_method == 'lowess':
        smooth_trend = sm.nonparametric.lowess(mean_trend, np.arange(len(mean_trend)), frac=0.1)[:, 1]
    else:
        smooth_trend = mean_trend

    # Plot the smoothed average trend
    plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

    # Optional shaded variability band
    # plt.fill_between(np.arange(len(mean_trend)), mean_trend - std_dev, mean_trend + std_dev, color='red', alpha=0.2)

    if y_range:
        plt.ylim(y_range)

    plt.ylabel(y_label, fontsize=50)
    plt.xlabel('Iteration', fontsize=50)

    num_iterations = all_particle_metrics.shape[1]
    tick_interval = max(1, num_iterations // 10)  # Show ~10 ticks

    tick_positions = np.arange(0, num_iterations, tick_interval)
    tick_labels = tick_positions + 1  # Labels should start from 1

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=50)

    plt.tick_params(axis='both', labelsize=50)
    plt.legend(fontsize=50, loc='upper right')
    plt.show()


# def get_best_particle_per_iteration(results_df, metric_column, agg_func='max'):
#     results = convert_results_df(results_df)
#     best_res = []

#     for i in range(len(results)):  # Each iteration
#         if agg_func == 'mean':
#             best = min([np.mean(results[i][j][metric_column]) for j in range(len(results[i]))])
#         elif agg_func == 'max':
#             best = min([np.max(results[i][j][metric_column]) for j in range(len(results[i]))])
#         else:
#             best = min([results[i][j][metric_column] for j in range(len(results[i]))])
#         best_res.append(best)

#     return best_res  # length = num_iterations

def get_best_particle_per_iteration(results_df, metric_column, agg_func='max'):
    results = convert_results_df(results_df)
    best_res = []
    best_clusters = []

    for iteration_particles in results:
        # Vectorized processing using list comprehensions
        if agg_func == 'mean':
            metrics = [np.mean(p[metric_column]) for p in iteration_particles]
        elif agg_func == 'max':
            metrics = [np.max(p[metric_column]) for p in iteration_particles]
        else:
            metrics = [p[metric_column] for p in iteration_particles]

        # Convert to NumPy array for efficient argmin
        metrics = np.array(metrics)
        best_idx = np.argmin(metrics)

        best_res.append(metrics[best_idx])
        best_clusters.append(iteration_particles[best_idx].get('num_clusters', None))

    return best_res, best_clusters

# def compute_best_so_far(results_df, metric_column, agg_func='max'):
#     best_res = get_best_particle_per_iteration(results_df, metric_column, agg_func)

#     best_so_far = [best_res[0]]
#     for val in best_res[1:]:
#         best_so_far.append(min(best_so_far[-1], val))

#     return best_so_far

def compute_best_so_far(results_df, metric_column, agg_func='max'):
    best_res, best_clusters = get_best_particle_per_iteration(results_df, metric_column, agg_func)

    best_so_far_vals = [best_res[0]]
    best_so_far_clusters = [best_clusters[0]]

    for i in range(1, len(best_res)):
        if best_res[i] < best_so_far_vals[-1]:
            best_so_far_vals.append(best_res[i])
            best_so_far_clusters.append(best_clusters[i])
        else:
            best_so_far_vals.append(best_so_far_vals[-1])
            best_so_far_clusters.append(best_so_far_clusters[-1])

    return best_so_far_vals, best_so_far_clusters

def plot_global_best_trend(results_df, metric_column, y_label, 
                           agg_func='max', smooth_method=None, 
                           window_size=5, y_range=None):
    best_res = get_best_particle_per_iteration(results_df, metric_column, agg_func)

    plt.figure(figsize=(50, 20))

    # Optional smoothing
    if smooth_method == 'moving_avg':
        trend = pd.Series(best_res).rolling(window=window_size, center=True).mean()
    elif smooth_method == 'lowess':
        trend = sm.nonparametric.lowess(best_res, np.arange(len(best_res)), frac=0.1)[:, 1]
    else:
        trend = best_res

    plt.plot(trend, linewidth=10, label='Best per iteration')

    if y_range:
        plt.ylim(y_range)

    plt.ylabel(y_label, fontsize=50)
    plt.xlabel('Iteration', fontsize=50)

    num_iterations = len(best_res)
    tick_interval = max(1, num_iterations // 10)  # Show ~10 ticks

    tick_positions = np.arange(0, num_iterations, tick_interval)
    tick_labels = tick_positions + 1  # Labels should start from 1

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=50)

    plt.tick_params(axis='both', labelsize=50)
    plt.legend(fontsize=50)
    plt.show()

def plot_global_best_so_far(results_df, metric_column, y_label, 
                            agg_func='max', y_range=None, output_path=None,show_plot=True):

    # Compute cumulative best-so-far trend
    global_best_so_far = compute_best_so_far(results_df, metric_column, agg_func)

    plt.figure(figsize=(50, 20))
    plt.plot(global_best_so_far, linewidth=10, label='Best-so-far')

    if y_range:
        plt.ylim(y_range)

    plt.ylabel(y_label, fontsize=50)
    plt.xlabel('Iteration', fontsize=50)

    num_iterations = len(global_best_so_far)
    tick_interval = max(1, num_iterations // 10)  # Show ~10 ticks

    tick_positions = np.arange(0, num_iterations, tick_interval)
    tick_labels = tick_positions + 1  # Labels should start from 1

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=50)

    plt.tick_params(axis='both', labelsize=50)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=1, fontsize=50, frameon=False)
    
    # Optional: Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_global_best_so_far_combined(results_df, 
                                     metric_column_1, label_1, 
                                     metric_column_2, label_2, 
                                     agg_func='max', y_range=None, 
                                     color_1='red', color_2='blue', output_path=None,
                                     show_plot=True):
    # Compute both trends
    trend_1 = compute_best_so_far(results_df, metric_column_1, agg_func)
    trend_2 = compute_best_so_far(results_df, metric_column_2, agg_func)

    plt.figure(figsize=(50, 20))
    plt.plot(trend_1, color=color_1, linewidth=10, label=label_1)
    plt.plot(trend_2, color=color_2, linewidth=10, label=label_2)

    if y_range:
        plt.ylim(y_range)

    plt.ylabel('Number of Violated Records', fontsize=50)
    plt.xlabel('Iteration', fontsize=50)

    num_iterations = len(trend_1)
    tick_interval = max(1, num_iterations // 10)
    tick_positions = np.arange(0, num_iterations, tick_interval)
    tick_labels = tick_positions + 1

    plt.xticks(ticks=tick_positions, labels=tick_labels, fontsize=50)
    plt.tick_params(axis='both', labelsize=50)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=50, frameon=False)
    
    # Optional: Save the plot if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()


def extract_info_from_filename(filename, pattern):
    match = re.match(pattern, filename)
    if match:
        return {
            'k': int(match.group(1)),
            'n_cluster': int(match.group(2)),
            'model': match.group(3)
        }
    else:
        return None



# def summarize_global_best_across_files(folder_path, pattern, metric_column, agg_func='max'):
#     summary = []

#     for file in os.listdir(folder_path):
#         if file.endswith('.csv'):
#             info = extract_info_from_filename(file, pattern)
#             if info:
#                 file_path = os.path.join(folder_path, file)
#                 results_df = pd.read_csv(file_path)
#                 best_so_far = compute_best_so_far(results_df, metric_column, agg_func)
#                 global_best = min(best_so_far)

#                 summary.append({
#                     'k': info['k'],
#                     'n_cluster': info['n_cluster'],
#                     'model': info['model'],
#                     'global_best': global_best
#                 })

#     return pd.DataFrame(summary)

def summarize_global_best_across_files(folder_path, pattern, metric_column, agg_func='max'):
    summary = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            info = extract_info_from_filename(file, pattern)
            if info:
                file_path = os.path.join(folder_path, file)
                results_df = pd.read_csv(file_path)

                best_so_far_vals, best_so_far_clusters = compute_best_so_far(results_df, metric_column, agg_func)
                global_best = min(best_so_far_vals)
                best_index = best_so_far_vals.index(global_best)
                best_cluster = best_so_far_clusters[best_index]

                summary.append({
                    'k': info['k'],
                    'model': info['model'],
                    'global_best': global_best,
                    'n_cluster': best_cluster  # From best-so-far tracking
                })

    return pd.DataFrame(summary)


def plot_model_performance(summary_df, baseline_dict, output_path=None,show_plot=True):

    models = summary_df['model'].unique()

    for model in models:

        print(f"Plotting for model: {model}")

        df_model = summary_df[summary_df['model'] == model]

        # Sort by n_cluster to ensure proper line plotting
        df_model = df_model.sort_values(by='n_cluster')

        plt.figure(figsize=(50, 20))
        plt.plot(df_model['n_cluster'], df_model['global_best'], marker='o', markersize=20, label='Proposed model', color='blue', linewidth=10)

        baseline = baseline_dict.get(model)
        if baseline is not None:
            plt.axhline(y=baseline, color='red', linestyle='--', label='Baseline',linewidth=10)

        # plt.title(f'Model: {model}', fontsize=50)
        plt.xlabel('Number of Clusters', fontsize=50)
        plt.ylabel('Loss Values', fontsize=50)
        plt.tick_params(axis='both', labelsize=50)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=50, frameon=False)
        plt.grid(True)
        plt.xticks(df_model['n_cluster'].unique())
        plt.tight_layout()
        
        # Optional: Save the plot if output_path is provided
        if output_path:
            # Create plot filenames
            filename = f"{model}_n_cluster_vs_global_best_loss.png"
            plot_path = os.path.join(output_path, filename)
            plt.savefig(plot_path, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()

def get_metric_values_of_global_best_particle(results_df, metric_column, agg_func=None):
    results = convert_results_df(results_df)

    global_best_value = float('inf')
    global_best_values = []

    for iteration in results:
        for particle in iteration:
            if metric_column not in particle:
                continue

            values = particle[metric_column]

            if agg_func == 'mean':
                summary_val = np.mean(values)
            elif agg_func == 'max':
                summary_val = np.max(values)
            else:
                summary_val = values  # assume single value
           
            if isinstance(summary_val, list) or isinstance(summary_val, np.ndarray):
                summary_val = min(summary_val)  # just in case

            if summary_val < global_best_value:
                global_best_value = summary_val
                global_best_values = values  # capture full list of values

    return global_best_values  # list or array of metric values



# def plot_metric_trend_for_each_particle(results_df, metric_column, y_label, smooth_method='moving_avg', window_size=5, y_range=None):
#     results = convert_results_df(results_df)  # Convert results

#     plt.figure(figsize=(50, 20))  # Adjust figure size

#     all_res = []  # Store all particle values per iteration

#     for j in range(len(results[0])):  # Iterate over particles
#         # res = [np.mean(results[i][j][metric_column]) for i in range(len(results))]  # Extract metric values
#         res = [np.max(results[i][j][metric_column]) for i in range(len(results))]
#         all_res.append(res)
#         plt.plot(res, alpha=0.6, linewidth=0.5)  # Plot all particles

#     # Convert to NumPy array for efficient calculations
#     all_res = np.array(all_res)
#     mean_trend = np.mean(all_res, axis=0)
#     std_dev = np.std(all_res, axis=0)  # Compute standard deviation for shading

#     # **Trend Smoothing**
#     if smooth_method == 'moving_avg':
#         smooth_trend = pd.Series(mean_trend).rolling(window=window_size, center=True).mean()
#     elif smooth_method == 'lowess':
#         smooth_trend = sm.nonparametric.lowess(mean_trend, np.arange(len(mean_trend)), frac=0.1)[:, 1]
#     else:
#         smooth_trend = mean_trend  # Default to mean if no smoothing is applied

#     # **Plot Mean with Smoothed Trend**
#     plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

#     # **Shaded Region for Variability**
#     #plt.fill_between(np.arange(len(mean_trend)), mean_trend - std_dev, mean_trend + std_dev, color='red', alpha=0.2)

#     # **Apply Y-axis range if provided**
#     if y_range is not None:
#         plt.ylim(y_range[0], y_range[1])

#     plt.ylabel(y_label, fontsize=50)  
#     plt.xlabel('Iteration', fontsize=50)  
#     plt.tick_params(axis='y', labelsize=50)
#     plt.tick_params(axis='x', labelsize=50)
#     plt.legend(fontsize=50, loc='upper right')

#     plt.show()

# def plot_global_best_trend(results_df, metric_column, y_label):
#     results = convert_results_df(results_df)  # Convert results

#     plt.figure(figsize=(50, 20))  # Adjust figure size

#     # best_res = [min([np.mean(results[i][j][metric_column]) for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values
#     best_res = [min([np.max(results[i][j][metric_column]) for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values
    
#     # smooth_trend = pd.Series(best_res).rolling(window=5, center=True).mean()

#     plt.plot(best_res, linewidth=10)  # Plot all particles
#     # plt.plot(smooth_trend, color='red', linewidth=10, linestyle='dashed', label="Smoothed Trend")

#     plt.ylabel(y_label, fontsize=50)  
#     plt.xlabel('Iteration', fontsize=50)  
#     plt.tick_params(axis='y', labelsize=50)
#     plt.tick_params(axis='x', labelsize=50)

#     plt.show()

# def plot_global_best_so_far(results_df, metric_column, y_label):
#     results = convert_results_df(results_df)  # Convert results

#     plt.figure(figsize=(50, 20))  # Adjust figure size

#     # best_res = [min([np.mean(results[i][j][metric_column]) for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values
#     best_res = [min([np.max(results[i][j][metric_column]) for j in range(len(results[i]))]) for i in range(len(results))]  # Extract best metric values

#     # Compute the global best found so far
#     global_best_so_far = [best_res[0]]  # Initialize with the first value
#     for i in range(1, len(best_res)):
#         global_best_so_far.append(min(global_best_so_far[-1], best_res[i]))  # Update with minimum so far

#     plt.plot(global_best_so_far, linewidth=10)  # Plot the global best trend

#     plt.ylabel(y_label, fontsize=50)  
#     plt.xlabel('Iteration', fontsize=50)  
#     plt.tick_params(axis='y', labelsize=50)
#     plt.tick_params(axis='x', labelsize=50)

#     plt.show()