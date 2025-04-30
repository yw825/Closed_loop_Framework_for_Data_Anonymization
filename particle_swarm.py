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
from collections import defaultdict


from constants import *
import utils
import model_train

# Function to compute numerical distance
def get_numeric_distance(df, NQIs, particle):
    # Convert the dataframe columns to a NumPy array for faster operations
    df_values = df[NQIs].values  # Shape: (num_rows, num_NQIs)
    
    # Shape of particle: (num_centroids, num_NQIs)
    # Broadcast the centroid values across all rows in df to calculate distances
    diffs = df_values[:, np.newaxis, :] - particle[:, :len(NQIs)]  # Shape: (num_rows, num_centroids, num_NQIs)
    squared_diffs = diffs ** 2  # Squared differences
    num_dist = np.sum(squared_diffs, axis=2)  # Sum over the NQIs (axis=2)

    return num_dist

# Function to compute categorical distance
def get_categorical_distance(df, CQIs, NQIs, particle):
    # Create a matrix of size (len(df), num_particles) for categorical distance
    categorical_dist = np.zeros((len(df), particle.shape[0]))

    # Extract categorical data from df
    categorical_data = df[CQIs].values

    # Extract the centroids for categorical data (for each particle)
    centroids = particle[:, len(NQIs):]  # Get the categorical columns from particle

    # Compare categorical values using broadcasting: (categorical_data != centroids) returns a matrix of True/False
    diffs = (categorical_data[:, None, :] != centroids[None, :, :]).astype(int)

    # Sum the differences for each row to get the categorical distance for each particle
    categorical_dist = np.sum(diffs, axis=2)

    return categorical_dist

# Function to compute the total distance
def get_total_distance(df, CQIs, NQIs, particle, gamma):
    numeric_distance = get_numeric_distance(df, NQIs, particle)
    categorical_distance = get_categorical_distance(df, CQIs, NQIs, particle)

    # Convert the distances into DataFrames for alignment by PatientIdentifier
    numeric_df = pd.DataFrame(numeric_distance, index=df.index)
    categorical_df = pd.DataFrame(categorical_distance, index=df.index)

    total_distance = numeric_df + gamma * categorical_df
    return total_distance

# Function to get the minimum distance and cluster assignment
def get_min_distance(df, CQIs, NQIs, particle, gamma):
    total_distance = get_total_distance(df, CQIs, NQIs, particle, gamma)
    # min_distance = np.min(total_distance, axis=1)
    cluster_assignment = np.argmin(total_distance, axis=1)
    return cluster_assignment # min_distance


# def get_anonymized_data(df, CQIs, NQIs, particle, gamma, k_val):
#     cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
#     df['cluster'] = cluster_assignment
    
#     anonymized_data = []
#     violating_records = []

#     for cluster_index in np.unique(cluster_assignment):
#         cluster_data = df[df['cluster'] == cluster_index].copy()
#         centroid_values = particle[cluster_index]

#         if len(cluster_data) < k_val:
#             # This cluster violates k-anonymity
#             violating_records.extend(cluster_data.index.tolist())

#         # Apply anonymization (even for non-violating clusters)
#         cluster_data[NQIs] = centroid_values[:len(NQIs)]
#         cluster_data[CQIs] = centroid_values[len(NQIs):]
#         anonymized_data.append(cluster_data)

#     anonymized_data = pd.concat(anonymized_data)

#     # Return both the data and the violating record indices for penalty handling
#     return anonymized_data, violating_records

# def calculate_k_constraint(anonymized_df, k, n_cluster):

#     # Count the number of records per cluster
#     num_records_per_cluster = anonymized_df['cluster'].value_counts()

#     # Identify clusters that violate the k constraint
#     violating_clusters = num_records_per_cluster[num_records_per_cluster < k]

#     # Calculate the total number of k-violations (sum the deficits)
#     total_k_violation = np.sum(k - violating_clusters)

#     return {
#         "k violation": total_k_violation,
#         "violating clusters": violating_clusters
#     }

def classify_clusters(df, k_val):
    clusters = df.groupby('cluster')
    valid_clusters = {}
    violated_clusters = {}

    for cluster_index, cluster_data in clusters:
        if len(cluster_data) >= k_val:
            valid_clusters[cluster_index] = cluster_data
        else:
            violated_clusters[cluster_index] = cluster_data

    return valid_clusters, violated_clusters

def split_valid_clusters(valid_clusters, k_val):
    retained_records = []
    excess_pool = []

    for idx, cluster_data in valid_clusters.items():
        retained = cluster_data.sample(n=k_val, random_state=42)
        excess = cluster_data.drop(retained.index)

        retained_records.append(retained)
        excess_pool.append(excess)

    return pd.concat(retained_records), pd.concat(excess_pool)

# def fix_violated_clusters(violated_clusters, excess_pool, k_val):
#     fixed_clusters = []
#     violating_records = []

#     for idx, cluster_data in violated_clusters.items():
#         n_missing = k_val - len(cluster_data)

#         if len(excess_pool) >= n_missing:
#             additional = excess_pool.sample(n=n_missing, random_state=42)
#             excess_pool = excess_pool.drop(additional.index)

#             # Set the cluster ID of additional records to match the violated cluster
#             additional = additional.copy()
#             additional['cluster'] = idx

#             new_cluster = pd.concat([cluster_data, additional])
#         else:
#             # Not enough records to fix the cluster — flag it
#             new_cluster = cluster_data
#             violating_records.extend(cluster_data.index.tolist())

#         # Ensure all records in the cluster have the correct cluster ID
#         new_cluster['cluster'] = idx
#         fixed_clusters.append(new_cluster)

#     return pd.concat(fixed_clusters), excess_pool, violating_records

def fix_violated_clusters(violated_clusters, excess_pool, k_val):
    fixed_clusters = []
    violating_records = []

    if not violated_clusters:
        return pd.DataFrame(), excess_pool, violating_records  # Nothing to fix

    for idx, cluster_data in violated_clusters.items():
        n_missing = k_val - len(cluster_data)

        if len(excess_pool) >= n_missing:
            additional = excess_pool.sample(n=n_missing, random_state=42)
            excess_pool = excess_pool.drop(additional.index)

            # Set the cluster ID of additional records to match the violated cluster
            additional = additional.copy()
            additional['cluster'] = idx

            new_cluster = pd.concat([cluster_data, additional])
        else:
            new_cluster = cluster_data
            violating_records.extend(cluster_data.index.tolist())

        new_cluster['cluster'] = idx
        fixed_clusters.append(new_cluster)

    return pd.concat(fixed_clusters), excess_pool, violating_records


def apply_centroids(df, particle, CQIs, NQIs):
    anonymized_data = []
    for cluster_index in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_index].copy()
        centroid_values = particle[cluster_index]
        cluster_data[NQIs] = centroid_values[:len(NQIs)]
        cluster_data[CQIs] = centroid_values[len(NQIs):]
        anonymized_data.append(cluster_data)
    return pd.concat(anonymized_data)


def get_adaptive_anonymized_data(df, CQIs, NQIs, particle, gamma, k_val):
    tracking_info = {}

    # Assign clusters using min distance
    cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
    df = df.copy()
    df['cluster'] = cluster_assignment

    tracking_info["num_clusters"] = len(np.unique(cluster_assignment))

    # Split into valid and violated clusters
    valid_clusters, violated_clusters = classify_clusters(df, k_val)
    tracking_info["num_valid_clusters"] = len(valid_clusters)
    tracking_info["num_violated_clusters"] = len(violated_clusters)
    tracking_info["num_violating_records_before_adjusting"] = sum(len(cluster) for cluster in violated_clusters.values())

    # Retain k from each valid cluster
    retained, excess_pool = split_valid_clusters(valid_clusters, k_val)
    tracking_info["num_retained_records"] = len(retained)
    tracking_info["num_excess_records"] = len(excess_pool)

    # Fix violated clusters using excess pool
    fixed, remaining_pool, violating_records = fix_violated_clusters(violated_clusters, excess_pool, k_val)
    if fixed is not None and not fixed.empty:
        tracking_info["num_fixed_clusters"] = len(fixed['cluster'].unique())
    else:
        tracking_info["num_fixed_clusters"] = 0
    tracking_info["num_used_excess"] = tracking_info["num_excess_records"] - len(remaining_pool)
    tracking_info["num_remaining_excess"] = len(remaining_pool)
    tracking_info["num_unfixed_clusters"] = tracking_info["num_violated_clusters"] - tracking_info["num_fixed_clusters"]
    tracking_info["num_total_violating_records_after_adjusting"] = len(violating_records)

    # Combine everything before anonymizing
    final_df = pd.concat([retained, fixed, remaining_pool])
    anonymized_df = apply_centroids(final_df, particle, CQIs, NQIs)

    return anonymized_df, violating_records, tracking_info


def initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster):

    particles = np.empty((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)

    # Generate random values for NQIs (numerical)
    for i, nqi in enumerate(NQIs):
        lower_bound = bounds[nqi]['lower_bound']
        upper_bound = bounds[nqi]['upper_bound']

        # Randomly generate values within bounds for each cluster (2 clusters)
        particles[:, :, i] = np.random.randint(lower_bound, upper_bound, size=(n_population, n_cluster))
        
    # Generate random values for CQIs (categorical)
    for i, cqi in enumerate(CQIs):
        unique_values = df[cqi].dropna().unique()

        # Randomly assign values for each cluster from the unique categorical values
        particles[:, :, len(NQIs) + i] = np.random.choice(unique_values, size=(n_population, n_cluster))

    return particles

def update_categorical_variables(particle_categorical, CQIs, centv, levels):

    # Ensure centv is a 2D array (n_particles, n_categories)
    centv = np.array(centv, dtype=float)
    
    # Saremi, S., Mirjalili, S., & Lewis, A. (2015). 
    # How important is a transfer function in discrete heuristic algorithms. 
    # Neural Computing and Applications, 26, 625-640.
    # Calculate the T value for each element
    T = np.abs(centv / np.sqrt(centv**2 + 1))

    # Generate random values for each particle
    rand = np.random.uniform(0, 1, size=particle_categorical.shape)

    # Compare rand with T for each element, determining whether to update the category
    mask = rand < T

    for i, cqi in enumerate(CQIs):
        random_choice = np.random.choice(list(levels[cqi].keys()), size=particle_categorical.shape[0:2])
        particle_categorical[:,:, i] = np.where(mask[:,:, i], random_choice, particle_categorical[:,:, i])

    return particle_categorical


def check_bound(particle_numeric, lower_bounds, upper_bounds, column_means):
    # # Ensure particle_numeric is a float type to perform comparisons
    # particle_numeric = np.array(particle_numeric, dtype=float)

    # Apply masks for out-of-bound values for each column
    for col_idx in range(particle_numeric.shape[2]):  # Iterate over columns
        mask_low = particle_numeric[:,:, col_idx] < lower_bounds[col_idx]
        mask_high = particle_numeric[:,:, col_idx] > upper_bounds[col_idx]

        # Replace out-of-bound values with the corresponding column mean
        particle_numeric[mask_low, col_idx] = column_means[col_idx]
        particle_numeric[mask_high, col_idx] = column_means[col_idx]

    return particle_numeric.astype(float)  # Convert back to integer values if needed


def update_particles_velocity_and_location(particles, n_population, centv, pbest, global_best, NQIs, CQIs, levels, bounds, nqi_means):
    uc = np.random.uniform(0, 0.01, size=(n_population, 1, 1))
    ud = np.random.uniform(0, 0.01, size=(n_population, 1, 1))
    c = 1 - uc - ud 

    centv = np.array(centv, dtype=float)
    centv[:,:,:len(NQIs)] = c * np.array(centv)[:,:,:len(NQIs)] + uc * (np.array(pbest)[:,:,:len(NQIs)] - np.array(particles)[:,:,:len(NQIs)]) + \
                        ud * (np.array(global_best)[:,:len(NQIs)] - np.array(particles)[:,:,:len(NQIs)])

    # Update numeric variables in particles based on the velocities
    particles = np.array(particles)
    particles[:,:,:len(NQIs)] = np.array(particles)[:,:,:len(NQIs)] + centv[:,:,:len(NQIs)]

    # Ensure particles stay within bounds
    lower_bounds = np.array([bounds[NQI]['lower_bound'] for NQI in NQIs])
    upper_bounds = np.array([bounds[NQI]['upper_bound'] for NQI in NQIs])
    # Apply check_bound function to all particles
    particles[:,:,:len(NQIs)] = check_bound(particles[:,:,:len(NQIs)], lower_bounds, upper_bounds, nqi_means)

    ########################################################################################################
    # Update categorical velocities

    l = len(NQIs)
    r = l + len(CQIs)
    global_best = np.array(global_best)
    pbest = np.array(pbest)
    centv[:,:, l:r] = c * centv[:,:, l:r] + uc * (np.where(pbest[:,:, l:r] == particles[:,:, l:r], 0, 1)) + \
                        ud * (np.where(global_best[:,l:r] == particles[:,:, l:r], 0, 1))       

    # Update categorical variables in particles
    particles[:,:, l:r] = update_categorical_variables(particles[:,:,l:r], CQIs, centv[:,:,l:r], levels)
    
    return particles, centv



def run_particle_swarm_experiment(df, models, param_combinations, NQIs, CQIs, n_population, 
                                  maxIter,n_bootstrap, bounds, levels, nqi_means, filedirectory):

    # all_results = []

    for param_comb in param_combinations:
        # Unpack parameters
        gamma, k_val, n_cluster_val, initial_violation_threshold, violation_decay_rate, penalty_weight = param_comb

        print(f"Running with k = {k_val}, n_cluster = {n_cluster_val},  initial_violation_threshold = {initial_violation_threshold}, violation_decay_rate = {violation_decay_rate}, penalty_weight = {penalty_weight}")

        for name, model in models:
            print(f"Training model: {name}")

            # Initialize storage for results
            results = []

            # Clean all memory before each model loop
            centv = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)
            fit = np.zeros(n_population)
            k_violation = np.zeros(n_population)

            accuracy_score = np.zeros((n_population, n_bootstrap))
            precision_score = np.zeros((n_population, n_bootstrap))
            recall_score = np.zeros((n_population, n_bootstrap))
            f1_score = np.zeros((n_population, n_bootstrap))
            auc_score = np.zeros((n_population, n_bootstrap))
            loss_score = np.zeros((n_population, n_bootstrap))
            tp_score = np.zeros((n_population, n_bootstrap))
            tn_score = np.zeros((n_population, n_bootstrap))
            fp_score = np.zeros((n_population, n_bootstrap))
            fn_score = np.zeros((n_population, n_bootstrap))

            # Initialize best solutions
            global_best_fit = float('inf')
            pbest_fit = np.full(n_population, np.inf)
            pbest = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)

            # Initialize particles
            particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster_val)

            for iteration in range(maxIter):

                print(f"Iteration: {iteration}")
                iteration_info = []

                # Update violation threshold
                violation_threshold = max(initial_violation_threshold - iteration * violation_decay_rate, 0)

                for i in range(n_population):
                    # Generate anonymized data
                    # anonymized_df = get_anonymized_data(df, CQIs, NQIs, particles[i], gamma)
                    anonymized_df, violating_records, tracking_info = get_adaptive_anonymized_data(df, CQIs, NQIs, particles[i], gamma, k_val)
                    # print(violating_records)

                    # Check k-anonymity constraint
                    # k_anonymity = calculate_k_constraint(anonymized_df, k_val, n_cluster_val)
                    k_violation[i] = len(violating_records)

                    # Encode categorical variables
                    anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

                    # Train ML model and get evaluation metrics
                    accuracies, precisions, recalls, f1_scores, aucs, losses, tps, tns, fps, fns = model_train.train_model_bootstrap(
                        anonymized_df_encoded, name, model, n_bootstrap
                    )

                    accuracy_score[i] = accuracies
                    precision_score[i] = precisions
                    recall_score[i] = recalls
                    f1_score[i] = f1_scores
                    auc_score[i] = aucs
                    loss_score[i] = losses
                    tp_score[i] = tps
                    tn_score[i] = tns
                    fp_score[i] = fps
                    fn_score[i] = fns

                    iteration_info.append({
                        "ML model": name,
                        "Iteration": iteration,
                        "Particle": i,
                        "Accuracy": accuracy_score[i],
                        "Precision": precision_score[i],
                        "Recall": recall_score[i],
                        "F1 score": f1_score[i],
                        "AUC": auc_score[i],
                        "Entropy-Loss": loss_score[i],
                        "TP": tp_score[i],
                        "TN": tn_score[i],
                        "FP": fp_score[i],
                        "FN": fn_score[i],
                        **tracking_info 
                    })

                    # Compute objective function
                    # normalized_k_violation = utils.normalize_data(k_violation[i], 0, 500)
                    excess_violation = max(0, len(violating_records) - violation_threshold)
                    penalty = penalty_weight * excess_violation
                    fit[i] = np.max(loss_score[i]) + penalty
                    # fit[i] = np.mean(loss_score[i]) + penalty

                    # Update personal best
                    if fit[i] < pbest_fit[i]:
                        pbest_fit[i] = fit[i]
                        pbest[i] = particles[i]

                results.append(iteration_info)

                # Update global best
                if global_best_fit > min(fit):
                    global_best_fit = min(fit)
                    global_best = particles[np.argmin(fit)]

                # Update particles
                particles, centv = update_particles_velocity_and_location(
                    particles, n_population, centv, pbest, global_best, NQIs, CQIs, levels, bounds, nqi_means
                )

            # Save the best anonymized dataset
            best_anonymized_df = get_adaptive_anonymized_data(df, CQIs, NQIs, global_best, gamma, k_val)[0]

            filename = f"best_anonymized_df_k{k_val}_ncluster{n_cluster_val}.csv"
            filepath = os.path.join(filedirectory, filename)
            best_anonymized_df.to_csv(filepath, index=False)

            print(f"Saved the best anonymized data to {filepath}")

            # all_results.append(results)

            # Clean up memory
            del particles, centv, fit, k_violation, pbest, pbest_fit, global_best_fit, global_best
            del accuracy_score, precision_score, recall_score, f1_score, auc_score, loss_score, tp_score, tn_score, fp_score, fn_score
            # del anonymized_df, anonymized_df_encoded
            # del iteration_info, results
            # del best_anonymized_df
            # del filename, filepath
            # Run garbage collection to free up memory
            gc.collect()

    return results # all_results
