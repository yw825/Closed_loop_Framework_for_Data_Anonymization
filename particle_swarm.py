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
from sklearn.base import clone
import copy
from scipy.stats import entropy as scipy_entropy
import gower

from constants import *
import utils
import model_train
import concurrent.futures

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

def satisfies_k_anonymity(cluster_df, k_val):
    return len(cluster_df) >= k_val

def entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-12))  # add epsilon to avoid log(0)

def satisfies_l_diversity(cluster_df, SAs, l=2, check_each_sa=True, check_composite=False, composite_strict=False):
    log_l = np.log2(l)
    satisfies_l = True
    
    if check_each_sa:
        sa_entropies = cluster_df[SAs].apply(entropy)
        if not (sa_entropies >= log_l).all():
            satisfies_l = False

    if check_composite:
        composite_sa = cluster_df[SAs].astype(str).agg('|'.join, axis=1)
        if composite_strict:
            if entropy(composite_sa) < log_l:
                satisfies_l = False
        else:
            if composite_sa.nunique() < l:
                satisfies_l = False

    return satisfies_l

def classify_clusters(df, k_val, SAs):
    clusters = df.groupby('cluster')

    violates_both = []
    violates_k_only = []
    violates_l_only = []
    valid_clusters = []

    for cluster_index, cluster_data in clusters:
        satisfies_k = satisfies_k_anonymity(cluster_data, k_val)
        satisfies_l = satisfies_l_diversity(cluster_data, SAs)

        if not satisfies_k and not satisfies_l:
            violates_both.append(cluster_data)
        elif not satisfies_k:
            violates_k_only.append(cluster_data)
        elif not satisfies_l:
            violates_l_only.append(cluster_data)
        else:
            valid_clusters.append(cluster_data)

    return violates_both, violates_k_only, violates_l_only, valid_clusters

def split_valid_clusters(valid_clusters, k_val, SAs):
    retained_records = []
    excess_pool = []

    for cluster_data in valid_clusters:
        cluster_data = cluster_data.copy()
        found_valid_subset = False

        # Try all possible sizes from k to len(cluster)
        for size in range(k_val, len(cluster_data) + 1):
            subset = cluster_data.sample(n=size, random_state=42)
            satisfies_l = satisfies_l_diversity(subset, SAs)

            if satisfies_k_anonymity(subset, k_val) and satisfies_l:
                retained_records.append(subset)
                remaining = cluster_data.drop(subset.index)
                excess_pool.append(remaining)
                found_valid_subset = True
                break

        if not found_valid_subset:
            # If we couldn't find a valid subset (shouldn't happen if originally valid), retain all
            retained_records.append(cluster_data)

    # retained_df = pd.concat(retained_records, ignore_index=True)
    # excess_df = pd.concat(excess_pool, ignore_index=True) if excess_pool else pd.DataFrame()

    return retained_records, excess_pool

def get_centroid_values_from_particle(particle, cluster_list):
    centroid_info = []
    for cluster in cluster_list:
        if len(cluster) == 0:
            continue
        cluster_index = cluster['cluster'].values[0]
        centroid_vector = particle[cluster_index, :]
        centroid_info.append((centroid_vector, cluster_index))
    return centroid_info  # List of (vector, index)


def find_closest_centroid_to_pool(particle, excess_pool, violates_both, violates_k_only, violates_l_only):
    
    # Get pool centroids and their cluster indices
    pool_info = get_centroid_values_from_particle(particle, excess_pool)
    pool_centroids = [vec for vec, idx in pool_info]
    pool_indices = [idx for vec, idx in pool_info]


    # Get violated centroids and their cluster indices
    violated_centroids = []
    violated_indices = []
    for violated in [violates_both, violates_k_only, violates_l_only]:
        if len(violated) != 0:
            info = get_centroid_values_from_particle(particle, violated)
            violated_centroids.extend([vec for vec, idx in info])
            violated_indices.extend([idx for vec, idx in info])

    # Convert to DataFrames for Gower distance
    pool_df = pd.DataFrame(pool_centroids)
    violated_df = pd.DataFrame(violated_centroids)

    # Compute Gower distance matrix
    pool_distances = gower.gower_matrix(violated_df, pool_df)  # shape: (len(violated), len(pool))

    # Find closest pool centroid for each violated cluster
    # Create a grid of all combinations
    violated_grid, pool_grid = np.meshgrid(violated_indices, pool_indices, indexing='ij')

    # Flatten the arrays
    violated_flat = violated_grid.ravel()
    pool_flat = pool_grid.ravel()
    dist_flat = pool_distances.ravel()

    # Construct DataFrame
    distance_df = pd.DataFrame({
        'violated_cluster': violated_flat,
        'pool_cluster': pool_flat,
        'distance': dist_flat
    })

    # Sort by distance ascending
    sorted_distances = distance_df.sort_values(by=['violated_cluster', 'distance'], ascending=[True, True]).reset_index(drop=True)

    return sorted_distances

def fix_violated_clusters(violates_both, violates_k_only, violates_l_only, valid_clusters, SAs, k_val, particle):
    sorted_distances = find_closest_centroid_to_pool(particle, valid_clusters, violates_both, violates_k_only, violates_l_only)

    # Step 1: Convert violated clusters into a dictionary using 'cluster' column as the key
    violated_clusters_dict = {
        int(df['cluster'].iloc[0]): df.copy() for df in violates_both + violates_k_only + violates_l_only
        if not df.empty and 'cluster' in df.columns
    }
    # print(f"📦 Total violated clusters to fix: {len(violated_clusters_dict)}")

    # Step 2: Convert excess_pool clusters into a dictionary using 'cluster' column as the key
    excess_pool_dict = {
        int(df['cluster'].iloc[0]): df.copy()
        for df in valid_clusters
        if not df.empty and 'cluster' in df.columns
    }
    # print(f"🎯 Initial excess pool clusters: {len(excess_pool_dict)}")

    # Dictionary to store fixed violated clusters
    fixed_clusters = {}
    unfixed_clusters = {}

    # Step 3: Iterate over each unique violated cluster ID
    for violated_id in sorted_distances['violated_cluster'].unique():
        # print(f"\n🔧 Fixing violated cluster: {violated_id}")
        violated_df = violated_clusters_dict[violated_id]

        # Step 4: Get sorted pool candidates for this violated cluster
        candidates = sorted_distances[sorted_distances['violated_cluster'] == violated_id]
        # print(f"➡️  Candidate pool clusters for violated {violated_id}: {candidates['pool_cluster'].tolist()}")

        fixed = False  # Track if the current violated cluster gets fixed

        # Step 5: Try to fix using nearest pool clusters
        for _, row in candidates.iterrows():
            pool_id = row['pool_cluster']
            # print(f"   🔍 Trying pool cluster: {pool_id}")

            if pool_id not in excess_pool_dict:
                # print(f"   ❌ Pool cluster {pool_id} not found in excess_pool_dict.")
                continue

            pool_df = excess_pool_dict[pool_id]
            if pool_df.empty:
                # print(f"   ⚠️ Pool cluster {pool_id} is empty.")
                continue

            # Step 6: Add records from this pool to violated cluster until it satisfies constraints
            move_count = 0
            satisfies_l = satisfies_l_diversity(violated_df, SAs)
            while not (satisfies_k_anonymity(violated_df, k_val) and satisfies_l):
                if pool_df.empty:
                    # print(f"   ⚠️ Pool cluster {pool_id} ran out of records while fixing cluster {violated_id}.")
                    break

                record_to_move = pool_df.sample(n=1, random_state=42)
                pool_df = pool_df.drop(record_to_move.index).reset_index(drop=True)
                violated_df = pd.concat([violated_df, record_to_move], ignore_index=True)
                move_count += 1

                satisfies_l = satisfies_l_diversity(violated_df, SAs)

            # print(f"   ✅ Moved {move_count} record(s) from pool {pool_id} to violated {violated_id}")

            # Step 7: Update dicts
            excess_pool_dict[pool_id] = pool_df
            violated_df['cluster'] = violated_id
            violated_clusters_dict[violated_id] = violated_df

            # Step 8: Check if fix is successful
            if satisfies_k_anonymity(violated_df, k_val) and satisfies_l_diversity(violated_df, SAs):
                # print(f"🎉 Successfully fixed cluster {violated_id} using pool {pool_id}")
                fixed_clusters[violated_id] = violated_df
                fixed = True
                break
            # else:
            #     print(f"❌ Cluster {violated_id} still not valid after trying pool {pool_id}")

        if not fixed:
            # print(f"⚠️ Could not fix violated cluster {violated_id} with any available pool clusters.")
            unfixed_clusters[violated_id] = violated_df

    updated_excess_pool = [df for df in excess_pool_dict.values() if not df.empty]
    # print(f"\n🔄 Remaining clusters in excess pool after fixing: {len(updated_excess_pool)}")
    # print(f"🚨 Unfixed violated clusters count: {len(unfixed_clusters)}")

    return fixed_clusters, updated_excess_pool, unfixed_clusters

def verify_valid_clusters(valid_clusters, updated_excess_pool, unfixed_clusters, k_val, SAs):
    # Step 1: Create a dictionary for valid clusters
    valid_dict = {}
    for df in valid_clusters:
        cluster_id = int(df['cluster'].iloc[0])
        if cluster_id in valid_dict:
            valid_dict[cluster_id] = pd.concat([valid_dict[cluster_id], df], ignore_index=True)
        else:
            valid_dict[cluster_id] = df.copy()
    # print(f"🔍 Valid clusters count: {len(valid_dict)}")

    # Step 2: Merge excess pool records back into respective clusters
    for df in updated_excess_pool:
        cluster_id = int(df['cluster'].iloc[0])
        if cluster_id in valid_dict:
            valid_dict[cluster_id] = pd.concat([valid_dict[cluster_id], df], ignore_index=True)
        else:
            valid_dict[cluster_id] = df.copy()

    # Step 3: Re-check validity of all clusters
    verified_valid_clusters = []
    updated_unfixed_clusters = unfixed_clusters.copy()  # avoid mutating original dict

    for cluster_id, df in valid_dict.items():
        satisfies_k = satisfies_k_anonymity(df, k_val)
        satisfies_l = satisfies_l_diversity(df, SAs)

        if satisfies_k and satisfies_l:
            verified_valid_clusters.append(df)
        else:
            updated_unfixed_clusters[cluster_id] = df
            # print(f"⚠️ Cluster {cluster_id} no longer satisfies k/l. Added to unfixed_clusters.")

    return verified_valid_clusters, updated_unfixed_clusters


def apply_centroids(df, particle, CQIs, NQIs):
    anonymized_data = []
    for cluster_index in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_index].copy()
        centroid_values = particle[cluster_index]
        cluster_data[NQIs] = centroid_values[:len(NQIs)]
        cluster_data[CQIs] = centroid_values[len(NQIs):]
        anonymized_data.append(cluster_data)
    return pd.concat(anonymized_data)


def get_adaptive_anonymized_data(df, CQIs, NQIs, particle, gamma, k_val, SAs):
    updated_unfixed_clusters_list = []
    tracking_info = {}

    # Assign clusters using min distance
    cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
    df = df.copy()
    df['cluster'] = cluster_assignment
    # print("------------------------------------------------------")
    # print("Original data:")
    # print(f"Number of CLUSTERS in the original data: ",len(np.unique(cluster_assignment)))

    # Split into valid and violated clusters
    violates_both, violates_k_only, violates_l_only, valid_clusters = classify_clusters(df, k_val, SAs)
    tracking_info["num_valid_clusters"] = len(valid_clusters)
    tracking_info["num_violates_k_only"] = len(violates_k_only)
    tracking_info["num_violates_l_only"] = len(violates_l_only)
    tracking_info["num_violates_both"] = len(violates_both)
    # print(f"Number of valid CLUSTERS: ", tracking_info["num_valid_clusters"])
    # print(f"Number of violated k only CLUSTERS: ", tracking_info["num_violates_k_only"])
    # print(f"Number of violated l only CLUSTERS: ", tracking_info["num_violates_l_only"])
    # print(f"Number of violated both CLUSTERS: ", tracking_info["num_violates_both"])

    tracking_info["num_records_valid"] = sum(len(cluster) for cluster in valid_clusters)
    tracking_info["num_records_violates_k_only"] = sum(len(cluster) for cluster in violates_k_only)
    tracking_info["num_records_violates_l_only"] = sum(len(cluster) for cluster in violates_l_only)
    tracking_info["num_records_violates_both"] = sum(len(cluster) for cluster in violates_both)
    tracking_info["total_num_violated_records_before_adjusting"] = tracking_info["num_records_violates_k_only"] + tracking_info["num_records_violates_l_only"] + tracking_info["num_records_violates_both"]
    # print(f"Number of records in valid clusters: ", tracking_info["num_records_valid"])
    # print(f"Number of records in violated k only clusters: ", tracking_info["num_records_violates_k_only"])
    # print(f"Number of records in violated l only clusters: ", tracking_info["num_records_violates_l_only"])
    # print(f"Number of records in violated both clusters: ", tracking_info["num_records_violates_both"])
    # print(f"Number of total records in violated clusters: ", tracking_info["total_num_violated_records_before_adjusting"])

    if tracking_info["num_records_valid"] == len(df):
        # No need to fix anything
        anonymized_df = apply_centroids(df, particle, CQIs, NQIs)
        still_violated = pd.DataFrame()
        tracking_info["total_num_violated_records_after_adjusting"] = len(still_violated)
        tracking_info["num_clusters"] = len(np.unique(anonymized_df['cluster']))
    else:
        # Retain k and l from each valid cluster
        retained, excess_pool = split_valid_clusters(valid_clusters, k_val, SAs)
        # tracking_info["num_retained_clusters"] = len(retained)
        # tracking_info["num_excess_clusters"] = len(excess_pool)
        # print("------------------------------------------------------")
        # print("Splitting valid clusters starts here:")
        # print(f"num_retained_CLUSTERS: ", len(retained))
        # print(f"num_excess_CLUSTERS: ", len(excess_pool))

        # Fix violated clusters 
        fixed, remaining_pool, still_violated = fix_violated_clusters(violates_both, violates_k_only, violates_l_only, excess_pool, SAs, k_val, particle)
        # tracking_info["num_fixed_clusters"] = len(fixed)
        # tracking_info["num_unfixed_clusters"] = len(still_violated)
        # print("------------------------------------------------------")
        # print("Fixing violated clusters starts here:")
        # print("num_fixed_CLUSTERS: ", len(fixed))
        # print("num_CLUSTERS_left_in_the_pool: ", len(remaining_pool))
        # print("num_CLUSTERS_still_violated: ", len(still_violated))

        # Verify valid clusters are still valid after adjusting
        verified_valid_clusters, updated_unfixed_clusters = verify_valid_clusters(retained, remaining_pool, still_violated, k_val, SAs)
        tracking_info["num_fixed_clusters"] = len(verified_valid_clusters) 
        tracking_info["num_unfixed_clusters"] = len(updated_unfixed_clusters)
        tracking_info["total_num_violated_records_after_adjusting"] = sum(len(cluster) for cluster in updated_unfixed_clusters.values())
        # print("------------------------------------------------------")
        # print("Verifying if valid clusters are still valid:")
        # print(f"num_fixed_CLUSTERS: ", tracking_info["num_fixed_clusters"])
        # print(f"num_unfixed_CLUSTERS: ", tracking_info["num_unfixed_clusters"])
        # print(f"total_num_violated_records_after_adjusting", tracking_info["total_num_violated_records_after_adjusting"])


        # Combine everything before anonymizing
        fixed_clusters_list = list(fixed.values())
        updated_unfixed_clusters_list = list(updated_unfixed_clusters.values()) 
        final_df = pd.concat(fixed_clusters_list+verified_valid_clusters+updated_unfixed_clusters_list, ignore_index=True)
        # Anonymize the data
        anonymized_df = apply_centroids(final_df, particle, CQIs, NQIs)
        tracking_info["num_clusters"] = len(np.unique(anonymized_df['cluster']))
        # print("------------------------------------------------------")
        # print("Anonymization finished here:")
        # print("Number of CLUSTERS in anonymized data: ", tracking_info["num_clusters"])

    return anonymized_df, tracking_info, updated_unfixed_clusters_list


def initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster):

    particles = np.empty((n_population, n_cluster, len(NQIs) + len(CQIs)), dtype=object)

    # Generate random values for NQIs (numerical)
    for i, nqi in enumerate(NQIs):
        lower_bound = bounds[nqi]['lower_bound']
        upper_bound = bounds[nqi]['upper_bound']

        # Randomly generate values within bounds for each cluster (2 clusters)
        particles[:, :, i] = np.random.randint(lower_bound, upper_bound, size=(n_population, n_cluster))
        # particles[:, :, i] = np.random.uniform(lower_bound, upper_bound, size=(n_population, n_cluster))
        
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

# def evaluate_particle(i, particles, df, CQIs, NQIs, gamma, k_val, SAs, name, model, n_bootstrap,
#                       initial_violation_threshold, violation_decay_rate, penalty_weight,
#                       iteration, aggregate_function, levels, bounds, nqi_means):
#     anonymized_df, tracking_info, violating_records = get_adaptive_anonymized_data(
#         df, CQIs, NQIs, particles[i], gamma, k_val, SAs
#     )
#     anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

#     accuracies, precisions, recalls, f1_scores, aucs, losses, tps, tns, fps, fns = model_train.train_model_bootstrap(
#         anonymized_df_encoded, name, clone(model), n_bootstrap
#     )

#     excess_violation = max(0, len(violating_records) - max(initial_violation_threshold - iteration * violation_decay_rate, 0))
#     penalty = penalty_weight * excess_violation

#     if aggregate_function == 'mean':
#         fitness = np.mean(losses) + penalty
#     elif aggregate_function == 'max':
#         fitness = np.max(losses) + penalty
#     else:
#         raise ValueError(f"Unknown aggregate function: {aggregate_function}")

#     result = {
#         "index": i,
#         "fitness": fitness,
#         "particle": particles[i],
#         "tracking_info": tracking_info,
#         "scores": {
#             "Accuracy": accuracies,
#             "Precision": precisions,
#             "Recall": recalls,
#             "F1 score": f1_scores,
#             "AUC": aucs,
#             "Entropy-Loss": losses,
#             "TP": tps,
#             "TN": tns,
#             "FP": fps,
#             "FN": fns
#         }
#     }

#     return result

# def run_particle_swarm_experiment(df, name, model, gamma, k_val, SAs, n_cluster_val,
#                                    initial_violation_threshold, violation_decay_rate, penalty_weight,
#                                    NQIs, CQIs, n_population, maxIter, n_bootstrap,
#                                    bounds, levels, nqi_means, filedirectory, current_iter,
#                                    aggregate_function=None):

#     results = []

#     centv = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)
#     fit = np.zeros(n_population)
#     pbest_fit = np.full(n_population, np.inf)
#     pbest = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)

#     particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster_val)

#     for iteration in range(maxIter):
#         print(f"Iteration: {iteration}")
#         iteration_info = []

#         with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#             futures = [
#                 executor.submit(
#                     evaluate_particle, i, particles, df, CQIs, NQIs, gamma, k_val, SAs,
#                     name, model, n_bootstrap, initial_violation_threshold,
#                     violation_decay_rate, penalty_weight, iteration, aggregate_function,
#                     levels, bounds, nqi_means
#                 ) for i in range(n_population)
#             ]
#             particle_results = [f.result() for f in concurrent.futures.as_completed(futures)]

#         for result in particle_results:
#             i = result['index']
#             fit[i] = result['fitness']
#             if fit[i] < pbest_fit[i]:
#                 pbest_fit[i] = fit[i]
#                 pbest[i] = result['particle']

#             iteration_info.append({
#                 "ML model": name,
#                 "Iteration": iteration,
#                 "Particle": i,
#                 "Particle centroid": result['particle'],
#                 **result['tracking_info'],
#                 **result['scores']
#             })

#         results.append(copy.deepcopy(iteration_info))

#         if np.min(fit) < np.min(pbest_fit):
#             global_best_fit = np.min(fit)
#             global_best = particles[np.argmin(fit)]

#         particles, centv = update_particles_velocity_and_location(
#             particles, n_population, centv, pbest, global_best, NQIs, CQIs, levels, bounds, nqi_means
#         )

#     best_anonymized_df = get_adaptive_anonymized_data(df, CQIs, NQIs, global_best, gamma, k_val, SAs)[0]
#     filename = f"best_anonymized_df_k{k_val}_ncluster{n_cluster_val}_{name}_round{current_iter}.csv"
#     filepath = os.path.join(filedirectory, filename)
#     best_anonymized_df.to_csv(filepath, index=False)
#     print(f"Saved the best anonymized data to {filepath}")

#     gc.collect()
#     return results

def run_particle_swarm_experiment(df, name, model, gamma, k_val, SAs, n_cluster_val,initial_violation_threshold, violation_decay_rate, penalty_weight,
NQIs, CQIs, n_population, maxIter, n_bootstrap, bounds, levels, nqi_means, filedirectory, current_iter, aggregate_function=None):

    # Initialize storage for results
    results = []

    # Initialize variables
    centv = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)
    fit = np.zeros(n_population)
    # k_violation = np.zeros(n_population)

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

        # print(f"Iteration: {iteration}")
        iteration_info = []

        # Update violation threshold
        violation_threshold = max(initial_violation_threshold - iteration * violation_decay_rate, 0)

        for i in range(n_population):
            # Generate anonymized data
            # anonymized_df = get_anonymized_data(df, CQIs, NQIs, particles[i], gamma)
            anonymized_df, tracking_info, violating_records = get_adaptive_anonymized_data(df, CQIs, NQIs, particles[i], gamma, k_val, SAs)
            # print(f"Iteration {iteration}, Particle {i}: Data hash = {hash(pd.util.hash_pandas_object(anonymized_df).sum())}")

            # Check k-anonymity constraint
            # k_anonymity = calculate_k_constraint(anonymized_df, k_val, n_cluster_val)
            # k_violation[i] = len(violating_records)

            # Encode categorical variables
            anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

            # Train ML model and get evaluation metrics
            accuracies, precisions, recalls, f1_scores, aucs, losses, tps, tns, fps, fns  = model_train.train_model_bootstrap(
                anonymized_df_encoded, name, clone(model), n_bootstrap
            )
            # tps, tns, fps, fns           

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
            # print(f"Iteration {iteration}, Particle {i}: Losses = {loss_score[i]}")

            # del accuracies, precisions, recalls, f1_scores, aucs, losses, tps, tns, fps, fns

            iteration_info.append({
                "ML model": name,
                "Iteration": iteration,
                "Particle": i,
                "Particle centroid": particles[i],
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
            # print(f"In iteration_info, Iteration {iteration} Particel {i} results: {iteration_info[-1]['Entropy-Loss']}")

            # Compute objective function
            # normalized_k_violation = utils.normalize_data(k_violation[i], 0, 500)
            excess_violation = max(0, len(violating_records) - violation_threshold)
            penalty = penalty_weight * excess_violation
            # fit[i] = losses + penalty
            # print('Maximum loss score:', np.max(loss_score[i]))
            if aggregate_function == 'mean':
                fit[i] = np.mean(loss_score[i]) + penalty
            elif aggregate_function == 'max':
                fit[i] = np.max(loss_score[i]) + penalty
            else:
                raise ValueError(f"Unknown aggregate function: {aggregate_function}")
            # fit[i] = np.mean(loss_score[i]) + penalty

            # Update personal best
            if fit[i] < pbest_fit[i]:
                pbest_fit[i] = fit[i]
                pbest[i] = particles[i]

        results.append(copy.deepcopy(iteration_info))

        # for a in range(len(results)):
        #     print(f"In results, Iteration {a}:")
        #     for entry in results[a]:
        #         print(f"Particle {entry['Particle']}: Entropy-Loss = {entry['Entropy-Loss']}")


        # Update global best
        if global_best_fit > min(fit):
            global_best_fit = min(fit)
            global_best = particles[np.argmin(fit)]

        # Update particles
        particles, centv = update_particles_velocity_and_location(
            particles, n_population, centv, pbest, global_best, NQIs, CQIs, levels, bounds, nqi_means
        )

    # Save the best anonymized dataset
    best_anonymized_df = get_adaptive_anonymized_data(df, CQIs, NQIs, global_best, gamma, k_val, SAs)[0]

    filename = f"best_anonymized_df_k{k_val}_ncluster{n_cluster_val}_{name}_round{current_iter}.csv"
    filepath = os.path.join(filedirectory, filename)
    best_anonymized_df.to_csv(filepath, index=False)

    print(f"Saved the best anonymized data to {filepath}")

    # all_results.append(results)

    # Clean up memory
    del particles, centv, fit, pbest, pbest_fit, global_best_fit, global_best
    del accuracy_score, precision_score, recall_score, f1_score, auc_score, loss_score, tp_score, tn_score, fp_score, fn_score
    del anonymized_df, anonymized_df_encoded
    del best_anonymized_df
    # del filename, filepath
    # Run garbage collection to free up memory
    gc.collect()

    return results # all_results

def run_single_experiment(i, df, name, model, gamma, k_val, n_cluster_val,
                           initial_violation_threshold, violation_decay_rate,
                           penalty_weight, SAs, NQIs, CQIs, bounds, levels,
                           nqi_means, base_path):
    
    # print(f"PSO Round: {i}")
    results = particle_swarm.run_particle_swarm_experiment(
        df=df, 
        name=name,
        model=model, 
        gamma=gamma,
        k_val=k_val,
        SAs=SAs, 
        n_cluster_val=n_cluster_val,
        initial_violation_threshold=initial_violation_threshold,
        violation_decay_rate=violation_decay_rate,
        penalty_weight=penalty_weight, 
        NQIs=NQIs, 
        CQIs=CQIs, 
        n_population=10,
        maxIter=20,
        n_bootstrap=100,
        bounds=bounds, 
        levels=levels, 
        nqi_means=nqi_means, 
        filedirectory=os.path.join(base_path, 'Anonymized Data'),
        current_iter=i,
        aggregate_function='mean'
    )

    results_df = pd.DataFrame(results)
    filedirectory = os.path.join(base_path, 'Tracking Info')
    filename = f"track_info_k{k_val}_ncluster{n_cluster_val}_{name}_round{i}.csv"
    filename = os.path.join(filedirectory, filename)
    results_df.to_csv(filename, index=False)

def cluster_classification_test(df, gamma, k_val, n_cluster_val,
                                           NQIs, CQIs, n_population, bounds, SAs):
    # Initialize particles
    particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster_val)


    for i in range(n_population):
        print(f"Testing particle {i}")
        print(f"Particle shape: {particles[i].shape}")

        # Assign clusters using min distance
        cluster_assignment = get_min_distance(df, CQIs, NQIs, particles[i], gamma)
        df = df.copy()
        df['cluster'] = cluster_assignment

        # print(df['cluster'].isna().sum(), "records have NaN cluster IDs")

        # Split into valid and violated clusters
        violates_both, violates_k_only, violates_l_only, valid_clusters = classify_clusters(df, k_val, SAs)

        # # Print the number of clusters
        # print(f"num_valid_clusters: ", len(valid_clusters))
        # print(f"num_violates_k_only: ", len(violates_k_only))
        # print(f"num_violates_l_only: ", len(violates_l_only))
        # print(f"num_violates_both: ", len(violates_both))

        # print(f"num_records_valid: ", sum(len(cluster) for cluster in valid_clusters))
        # print(f"num_records_violates_k_only: ", sum(len(cluster) for cluster in violates_k_only))
        # print(f"num_records_violates_l_only: ", sum(len(cluster) for cluster in violates_l_only))
        # print(f"num_records_violates_both: ", sum(len(cluster) for cluster in violates_both))
        # print(f"total_num_violated_records_before_adjusting: ", sum(len(cluster) for cluster in violates_k_only) + sum(len(cluster) for cluster in violates_l_only) + sum(len(cluster) for cluster in violates_both))

        # # Retain k and l from each valid cluster
        # retained, excess_pool = split_valid_clusters(valid_clusters, k_val, SAs)

        # print(f"num_retained_clusters: ", len(retained))
        # print(f"num_excess_pool_clusters: ", len(excess_pool))
        # print(f"num_retained_records: ", sum(len(cluster) for cluster in retained))
        # print(f"num_excess_records: ", sum(len(cluster) for cluster in excess_pool))

        Ano, track, updated_unfixed_clusters_list = get_adaptive_anonymized_data(df, CQIs, NQIs, particles[i], gamma, k_val, SAs)

    return violates_both, violates_k_only, violates_l_only, valid_clusters, particles, Ano, track, updated_unfixed_clusters_list