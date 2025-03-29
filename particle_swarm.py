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
    min_distance = np.min(total_distance, axis=1)
    cluster_assignment = np.argmin(total_distance, axis=1)
    return min_distance, cluster_assignment

# Main function to anonymize data based on closest cluster
def get_anonymized_data(df, CQIs, NQIs, particle, gamma):
    min_distance, cluster_assignment = get_min_distance(df, CQIs, NQIs, particle, gamma)
    df['cluster'] = cluster_assignment
    
    anonymized_data = []
    for cluster_index in np.unique(cluster_assignment):
        cluster_data = df[df['cluster'] == cluster_index].copy()
        centroid_values = particle[cluster_index]
        
        # Update numeric and categorical values with the centroid
        cluster_data[NQIs] = centroid_values[:len(NQIs)]
        cluster_data[CQIs] = centroid_values[len(NQIs):]
        
        anonymized_data.append(cluster_data)
    
    anonymized_data = pd.concat(anonymized_data)
    return anonymized_data


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



def calculate_k_constraint(anonymized_df, k, n_cluster):

    # Count the number of records per cluster
    num_records_per_cluster = anonymized_df['cluster'].value_counts()

    # Identify clusters that violate the k constraint
    violating_clusters = num_records_per_cluster[num_records_per_cluster < k]

    # Calculate the total number of k-violations (sum the deficits)
    total_k_violation = np.sum(k - violating_clusters)

    return {
        "k violation": total_k_violation,
        "violating clusters": violating_clusters
    }



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
        gamma, k_val, n_cluster_val, l_multi_k_val, l_multi_ML_val = param_comb

        print(f"Running with k = {k_val}, n_cluster = {n_cluster_val}, l_multi_k = {l_multi_k_val}, l_multi_ML = {l_multi_ML_val}")

        for name, model in models:
            print(f"Training model: {name}")

            # Initialize storage for results
            results = []

            # Clean all memory before each model loop
            centv = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)
            fit = np.zeros(n_population)
            k_violation = np.zeros(n_population)

            accuracy_score = np.zeros(n_population)
            precision_score = np.zeros(n_population)
            recall_score = np.zeros(n_population)
            f1_score = np.zeros(n_population)
            auc_score = np.zeros(n_population)
            loss_score = np.zeros(n_population)
            confusion_matrix = np.zeros((n_population, 2, 2))

            # Initialize best solutions
            global_best_fit = float('inf')
            pbest_fit = np.full(n_population, np.inf)
            pbest = np.zeros((n_population, n_cluster_val, len(NQIs) + len(CQIs)), dtype=object)

            # Initialize particles
            particles = initialize_particles(n_population, NQIs, CQIs, bounds, df, n_cluster_val)

            for iteration in range(maxIter):
                print(f"Iteration: {iteration}")
                iteration_info = []

                for i in range(n_population):
                    # Generate anonymized data
                    anonymized_df = get_anonymized_data(df, CQIs, NQIs, particles[i], gamma)

                    # Check k-anonymity constraint
                    k_anonymity = calculate_k_constraint(anonymized_df, k_val, n_cluster_val)
                    k_violation[i] = k_anonymity['k violation']

                    # Encode categorical variables
                    anonymized_df_encoded = utils.encode_categorical_from_file(anonymized_df)

                    # Train ML model and get evaluation metrics
                    avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_auc, avg_loss, avg_cm = model_train.train_model_bootstrap(
                        anonymized_df_encoded, model, n_bootstrap
                    )

                    accuracy_score[i] = avg_accuracy
                    precision_score[i] = avg_precision
                    recall_score[i] = avg_recall
                    f1_score[i] = avg_f1_score
                    auc_score[i] = avg_auc
                    loss_score[i] = avg_loss
                    confusion_matrix[i] = avg_cm

                    iteration_info.append({
                        "ML model": name,
                        "Iteration": iteration,
                        "Particle": i,
                        "k violation": k_violation[i],
                        "Accuracy": avg_accuracy,
                        "Precision": avg_precision,
                        "Recall": avg_recall,
                        "F1 score": avg_f1_score,
                        "AUC": avg_auc,
                        "Entropy-Loss": avg_loss,
                        "Confusion matrix": avg_cm
                    })

                    # Compute objective function
                    normalized_k_violation = utils.normalize_data(k_violation[i], 0, 500)
                    fit[i] = l_multi_k_val * normalized_k_violation + l_multi_ML_val * avg_loss

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
            best_anonymized_df = get_anonymized_data(df, CQIs, NQIs, global_best, gamma)

            filename = f"best_anonymized_df_k{k_val}_ncluster{n_cluster_val}_lmk{l_multi_k_val}_lmML{l_multi_ML_val}.csv"
            filepath = os.path.join(filedirectory, filename)
            best_anonymized_df.to_csv(filepath, index=True)

            print(f"Saved the best anonymized data to {filepath}")

            # all_results.append(results)

            # Clean up memory
            del particles, centv, fit, k_violation, pbest, pbest_fit, global_best_fit, global_best
            del accuracy_score, precision_score, recall_score, f1_score, auc_score, confusion_matrix
            gc.collect()

    return results # all_results
