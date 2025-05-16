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

# def classify_k_anonymity_violations(df, k_val):
#     clusters = df.groupby('cluster')
#     valid_clusters = {}
#     violated_clusters = {}

#     for cluster_index, cluster_data in clusters:
#         if len(cluster_data) >= k_val:
#             valid_clusters[cluster_index] = cluster_data
#         else:
#             violated_clusters[cluster_index] = cluster_data

#     return valid_clusters, violated_clusters

# def classify_l_diversity_violations(df, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False):
    # clusters = df.groupby('cluster')
    # valid_clusters = {}
    # violated_clusters = {}

    # log_l = np.log2(l)  # ℓ = 2

    # for cluster_index, cluster_data in clusters:
    #     individual_valid = True
    #     composite_valid = True

    #     # Individual SA entropy checks
    #     if check_each_sa:
    #         sa_entropies = cluster_data[SA].apply(entropy)
    #         individual_valid = (sa_entropies >= log_l).all()

    #     # Composite check (strict = entropy, loose = count)
    #     if check_composite:
    #         composite_sa = cluster_data[SA].astype(str).agg('|'.join, axis=1)

    #         if composite_strict:
    #             comp_entropy = entropy(composite_sa)
    #             if comp_entropy < log_l:
    #                 composite_valid = False
    #         else:
    #             if composite_sa.nunique() < l:
    #                 composite_valid = False

    #     # Classify
    #     if individual_valid and composite_valid:
    #         valid_clusters[cluster_index] = cluster_data
    #     else:
    #         violated_clusters[cluster_index] = cluster_data

    # return valid_clusters, violated_clusters

def satisfies_k_anonymity(cluster_df, k_val):
    return len(cluster_df) >= k_val

def entropy(series):
    probs = series.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs + 1e-12))  # add epsilon to avoid log(0)

def satisfies_l_diversity(cluster_df, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False):
    log_l = np.log2(l)
    satisfies_l = True

    if check_each_sa:
        sa_entropies = cluster_df[SA].apply(entropy)
        if not (sa_entropies >= log_l).all():
            satisfies_l = False

    if check_composite:
        composite_sa = cluster_df[SA].astype(str).agg('|'.join, axis=1)
        if composite_strict:
            if entropy(composite_sa) < log_l:
                satisfies_l = False
        else:
            if composite_sa.nunique() < l:
                satisfies_l = False

    return satisfies_l

def classify_clusters(df, k_val, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False):
    clusters = df.groupby('cluster')

    violates_both = {}
    violates_k_only = {}
    violates_l_only = {}
    valid_clusters = {}

    for cluster_index, cluster_data in clusters:
        satisfies_k = satisfies_k_anonymity(cluster_data, k_val)
        satisfies_l = satisfies_l_diversity(cluster_data, SA, l, check_each_sa, check_composite, composite_strict)

        if not satisfies_k and not satisfies_l:
            violates_both[cluster_index] = cluster_data
        elif not satisfies_k:
            violates_k_only[cluster_index] = cluster_data
        elif not satisfies_l:
            violates_l_only[cluster_index] = cluster_data
        else:
            valid_clusters[cluster_index] = cluster_data

    return violates_both, violates_k_only, violates_l_only, valid_clusters

def split_valid_clusters(valid_clusters, k_val, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False):
    retained_records = []
    excess_pool = []

    for idx, cluster_data in valid_clusters.items():
        cluster_data = cluster_data.copy()
        found_valid_subset = False

        # Try all possible sizes from k to len(cluster)
        for size in range(k_val, len(cluster_data) + 1):
            subset = cluster_data.sample(n=size, random_state=42)
            satisfies_l = satisfies_l_diversity(subset, SA, l, check_each_sa, check_composite, composite_strict)

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

def merge_violated_clusters(
    violated_clusters, k_val, SA, l=2,
    check_each_sa=True, check_composite=True, composite_strict=False
):
    fixed_clusters = []
    still_violated = []
    # Convert dictionary to list of DataFrames
    buffer = list(violated_clusters.values())

    while buffer:
        base = buffer.pop(0)
        # print(f"\n🔁 Starting new base cluster: {base['cluster'].iloc[0]} | Size: {len(base)}")
        # print(base['cluster'].isna().sum(), "records have NaN cluster IDs")

        while buffer:
            next_cluster = buffer.pop(0)
            # print(f"  ➕ Trying to merge with cluster: {next_cluster['cluster'].iloc[0]} | Size: {len(next_cluster)}")
            # print(next_cluster['cluster'].isna().sum(), "records have NaN cluster IDs")
            
            combined = pd.concat([base, next_cluster], ignore_index=True)

            satisfies_l = satisfies_l_diversity(
                combined, SA, l=l,
                check_each_sa=check_each_sa,
                check_composite=check_composite,
                composite_strict=composite_strict
            )

            # print(f"    🧪 Combined size: {len(combined)} | Satisfies k: {satisfies_k_anonymity(combined, k_val)}, Satisfies l: {satisfies_l}")

            if satisfies_k_anonymity(combined, k_val) and satisfies_l:
                # print("    ✅ Successfully formed valid cluster. Added to fixed_clusters.")
                # Unify the cluster ID using the most frequent one
                most_common_cluster_id = combined['cluster'].mode()[0]
                combined['cluster'] = most_common_cluster_id
                # print(combined['cluster'].isna().sum(), "records have NaN cluster IDs")
                # print(f"    🛠 Reassigned cluster ID to: {most_common_cluster_id}")
                fixed_clusters.append(combined)
                break
            else:
                # Update base and continue trying to merge with the next one
                base = combined
        else:
            # Inner loop exhausted without finding a satisfying merge
            satisfies_l = satisfies_l_diversity(
                base, SA, l=l,
                check_each_sa=check_each_sa,
                check_composite=check_composite,
                composite_strict=composite_strict
            )
            if satisfies_k_anonymity(base, k_val) and satisfies_l:
                # print("    ⚠️ Base satisfies both k and l after all merges. Added to fixed_clusters.")
                fixed_clusters.append(base)
            else:
                # print("    ❌ No valid merge found. Still violated. Sent to still_violated.")
                still_violated.append(base)

    return fixed_clusters, still_violated        
        
def fix_still_violated_clusters(violated_clusters, excess_pool, SA, k_val):
    fixed_cluster = []

    while excess_pool:
        base =  excess_pool.pop(0)
        # print(f"\n🔁 Starting new excess pool cluster: {base['cluster'].iloc[0]} | Size: {len(base)}")

        combined = pd.concat([violated_clusters, base], ignore_index=True)

        satisfies_l = satisfies_l_diversity(
            combined, SA, l=2,
            check_each_sa=True, check_composite=True, composite_strict=False
        )

        # print(f"    🧪 Combined size: {len(combined)} | Satisfies k: {satisfies_k_anonymity(combined, k_val)}, Satisfies l: {satisfies_l}")

        if satisfies_k_anonymity(combined, k_val) and satisfies_l:
            # print("    ✅ Successfully formed valid cluster. Added to fixed_clusters.")
            # Unify the cluster ID using the most frequent one
            most_common_cluster_id = combined['cluster'].mode()[0]
            combined['cluster'] = most_common_cluster_id
            fixed_cluster.append(combined)
            break
        else:
            # Update base and continue trying to merge with the next one
            base = combined
    return fixed_cluster, excess_pool  


def fix_all_violated_clusters(violates_both, violates_k_only, violates_l_only, excess_pool, k_val, SA, l_val=2):
    import pandas as pd

    fixed_clusters_list = []
    still_violated_list = []

    # Handle violates_k_only
    if violates_k_only:
        # print(f"\n🔍 Merging {len(violates_k_only)} clusters violating k only")
        fixed_k_clusters, still_violated_k = merge_violated_clusters(
            violates_k_only, k_val, SA, l=l_val,
            check_each_sa=True, check_composite=True, composite_strict=False
        )
        # print(f"✅ Fixed {len(fixed_k_clusters)} clusters | ❌ Still violated: {len(still_violated_k)}")
        fixed_clusters_list.extend(fixed_k_clusters)
        still_violated_list.extend(still_violated_k)

    # Handle violates_l_only
    if violates_l_only:
        # print(f"\n🔍 Merging {len(violates_l_only)} clusters violating l only")
        fixed_l_clusters, still_violated_l = merge_violated_clusters(
            violates_l_only, k_val, SA, l=l_val,
            check_each_sa=True, check_composite=True, composite_strict=False
        )
        # print(f"✅ Fixed {len(fixed_l_clusters)} clusters | ❌ Still violated: {len(still_violated_l)}")
        fixed_clusters_list.extend(fixed_l_clusters)
        still_violated_list.extend(still_violated_l)

    # Handle violates_both
    if violates_both:
        # print(f"\n🔍 Merging {len(violates_both)} clusters violating both k and l")
        fixed_both_clusters, still_violated_both = merge_violated_clusters(
            violates_both, k_val, SA, l=l_val,
            check_each_sa=True, check_composite=True, composite_strict=False
        )
        # print(f"✅ Fixed {len(fixed_both_clusters)} clusters | ❌ Still violated: {len(still_violated_both)}")
        fixed_clusters_list.extend(fixed_both_clusters)
        still_violated_list.extend(still_violated_both)

    # print(f"\n🔍 fixed_clusters size before handling still violated: {sum(len(cluster) for cluster in fixed_clusters_list)}")
    # print(f"🔍 still_violated size before handling still violated: {sum(len(cluster) for cluster in still_violated_list)}")

    fixed_clusters_all = []
    remaining_pool = []

    # Handle still_violated
    if still_violated_list:
        still_violated = pd.concat(still_violated_list, ignore_index=True)
        # print(f"\n⚠️ Final still_violated size: {len(still_violated)}")

        # Unify the cluster ID
        if 'cluster' in still_violated.columns:
            # Option 1: Use the most frequent cluster ID
            most_common_cluster_id = still_violated['cluster'].mode()[0]
        else:
            still_violated['cluster'] = -1
            most_common_cluster_id = -1

        still_violated['cluster'] = most_common_cluster_id
    else:
        still_violated = pd.DataFrame()

    satisfies_l = satisfies_l_diversity(
        still_violated, SA, l=l_val,
        check_each_sa=True, check_composite=True, composite_strict=False
    )

    if satisfies_k_anonymity(still_violated, k_val) and satisfies_l:
        # print("✅ All violations resolved.")
        fixed_clusters_list.append(still_violated)
        remaining_pool = excess_pool
        still_violated = pd.DataFrame()
    else:
        # print("❌ Some violations remain. Attempting to fill with excess_df.")
        fixed_clusters_all, remaining_pool = fix_still_violated_clusters(
            still_violated, excess_pool, SA, k_val
        )
        fixed_clusters_list.extend(fixed_clusters_all)
        still_violated = pd.DataFrame()

    # # Safe to use now
    # print(f"\n📦 Final fixed_clusters size after handling still violated: {sum(len(cluster) for cluster in fixed_clusters_all)}")
    # print(f"\n📦 Final remaining_pool size after handling still violated: {sum(len(cluster) for cluster in remaining_pool)}")

    # Concatenate lists of DataFrames into single DataFrames
    if fixed_clusters_list:
        fixed_clusters = pd.concat(fixed_clusters_list, ignore_index=True)
        # print(f"\n📦 Final fixed_clusters size after handling still violated: {len(fixed_clusters)}")
        remaining_pool = pd.concat(remaining_pool, ignore_index=True)
        # print(f"📦 Final remaining_pool size: {len(remaining_pool)}")
    else:
        fixed_clusters = pd.DataFrame()

    return fixed_clusters, remaining_pool, still_violated


# def fix_violated_clusters(violates_both, violates_k_only, violates_l_only, excess_df, k_val):
#     fixed_clusters = []
#     violating_records = []

#     if not violated_clusters:
#         return pd.DataFrame(), excess_pool, violating_records  # Nothing to fix

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
#             new_cluster = cluster_data
#             violating_records.extend(cluster_data.index.tolist())

#         new_cluster['cluster'] = idx
#         fixed_clusters.append(new_cluster)

#     return pd.concat(fixed_clusters), excess_pool, violating_records


def apply_centroids(df, particle, CQIs, NQIs):
    anonymized_data = []
    for cluster_index in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster_index].copy()
        centroid_values = particle[cluster_index]
        cluster_data[NQIs] = centroid_values[:len(NQIs)]
        cluster_data[CQIs] = centroid_values[len(NQIs):]
        anonymized_data.append(cluster_data)
    return pd.concat(anonymized_data)


def get_adaptive_anonymized_data(df, CQIs, NQIs, particle, gamma, k_val, SA):
    tracking_info = {}

    # Assign clusters using min distance
    cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
    df = df.copy()
    df['cluster'] = cluster_assignment

    # tracking_info["num_clusters"] = len(np.unique(cluster_assignment))

    # Split into valid and violated clusters
    violates_both, violates_k_only, violates_l_only, valid_clusters = classify_clusters(df, k_val, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False)
    # tracking_info["num_valid_clusters"] = len(valid_clusters)
    # tracking_info["num_violates_k_only"] = len(violates_k_only)
    # tracking_info["num_violates_l_only"] = len(violates_l_only)
    # tracking_info["num_violates_both"] = len(violates_both)

    tracking_info["num_records_valid"] = sum(len(cluster) for cluster in valid_clusters.values())
    tracking_info["num_records_violates_k_only"] = sum(len(cluster) for cluster in violates_k_only.values())
    tracking_info["num_records_violates_l_only"] = sum(len(cluster) for cluster in violates_l_only.values())
    tracking_info["num_records_violates_both"] = sum(len(cluster) for cluster in violates_both.values())
    tracking_info["total_num_violated_records_before_adjusting"] = tracking_info["num_records_violates_k_only"] + tracking_info["num_records_violates_l_only"] + tracking_info["num_records_violates_both"]

    # Retain k and l from each valid cluster
    retained, excess_pool = split_valid_clusters(valid_clusters, k_val, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False)
    retained = pd.concat(retained, ignore_index=True)
    tracking_info["num_retained_records"] = len(retained)
    # tracking_info["num_excess_records"] = sum(len(cluster) for cluster in excess_pool)

    # Fix violated clusters using excess pool
    fixed, remaining_pool, still_violated = fix_all_violated_clusters(violates_both, violates_k_only, violates_l_only, excess_pool, k_val, SA, l_val=2)
    tracking_info["num_fixed_records"] = len(fixed) 
    tracking_info["num_remaining_records"] = len(remaining_pool)
    tracking_info["total_num_violated_records_after_adjusting"] = len(still_violated)

    # Combine everything before anonymizing

    final_df = pd.concat([retained, fixed, remaining_pool, still_violated], ignore_index=True)
    anonymized_df = apply_centroids(final_df, particle, CQIs, NQIs)
    tracking_info["num_clusters"] = len(np.unique(anonymized_df['cluster']))

    return anonymized_df, tracking_info, still_violated


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



def run_particle_swarm_experiment(df, name, model, gamma, k_val, SA, n_cluster_val,initial_violation_threshold, violation_decay_rate, penalty_weight,
NQIs, CQIs, n_population, maxIter, n_bootstrap, bounds, levels, nqi_means, filedirectory,aggregate_function=None):

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

        print(f"Iteration: {iteration}")
        iteration_info = []

        # Update violation threshold
        violation_threshold = max(initial_violation_threshold - iteration * violation_decay_rate, 0)

        for i in range(n_population):
            # Generate anonymized data
            # anonymized_df = get_anonymized_data(df, CQIs, NQIs, particles[i], gamma)
            anonymized_df, tracking_info, violating_records = get_adaptive_anonymized_data(df, CQIs, NQIs, particles[i], gamma, k_val, SA)
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
    best_anonymized_df = get_adaptive_anonymized_data(df, CQIs, NQIs, global_best, gamma, k_val, SA)[0]

    filename = f"best_anonymized_df_k{k_val}_ncluster{n_cluster_val}_{name}.csv"
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


def cluster_classification_test(df, gamma, k_val, n_cluster_val,
                                           NQIs, CQIs, n_population, bounds, SA):
    # Initialize particles
    particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster_val)


    for i in range(n_population):
        print(f"Testing particle {i}")

        # Assign clusters using min distance
        cluster_assignment = get_min_distance(df, CQIs, NQIs, particles[i], gamma)
        df = df.copy()
        df['cluster'] = cluster_assignment

        print(df['cluster'].isna().sum(), "records have NaN cluster IDs")

        # Split into valid and violated clusters
        violates_both, violates_k_only, violates_l_only, valid_clusters = classify_clusters(df, k_val, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False)

        # # Print the number of clusters
        # print(f"num_valid_clusters: ", len(valid_clusters))
        # print(f"num_violates_k_only: ", len(violates_k_only))
        # print(f"num_violates_l_only: ", len(violates_l_only))
        # print(f"num_violates_both: ", len(violates_both))

        print(f"num_records_valid: ", sum(len(cluster) for cluster in valid_clusters.values()))
        print(f"num_records_violates_k_only: ", sum(len(cluster) for cluster in violates_k_only.values()))
        print(f"num_records_violates_l_only: ", sum(len(cluster) for cluster in violates_l_only.values()))
        print(f"num_records_violates_both: ", sum(len(cluster) for cluster in violates_both.values()))
        print(f"total_num_violated_records_before_adjusting: ", sum(len(cluster) for cluster in violates_k_only.values()) + sum(len(cluster) for cluster in violates_l_only.values()) + sum(len(cluster) for cluster in violates_both.values()))

        # Retain k and l from each valid cluster
        retained, excess_pool = split_valid_clusters(valid_clusters, k_val, SA, l=2, check_each_sa=True, check_composite=True, composite_strict=False)
        print(f"num_retained_records: ", sum(len(cluster) for cluster in retained))
        # tracking_info["num_excess_records"] = sum(len(cluster) for cluster in excess_pool)

        # Fix violated clusters using excess pool
        fixed, remaining_pool, still_violated = fix_all_violated_clusters(violates_both, violates_k_only, violates_l_only, excess_pool, k_val, SA, l_val=2)
        print(f"num_fixed_records: ", len(fixed))
        print(f"num_remaining_records: ", len(remaining_pool))
        print(f"total_num_violated_records_after_adjusting: ", len(still_violated))

    return violates_both, violates_k_only, violates_l_only, valid_clusters