import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Directories
OUTPUTS_DIR = r'C:\Users\wazus\OneDrive\Desktop\Evolutionary Computation ALL\outputs'
PLOTS_DIR = r'C:\Users\wazus\OneDrive\Desktop\Evolutionary Computation ALL\plots'

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

def read_solutions(instance_name):
    """Reads solutions from the CSV file for a given instance."""
    solutions = []
    csv_file = os.path.join(OUTPUTS_DIR, instance_name, 'GREEDY_TWO_EDGES_EXCHANGE_RandomStart_all_solutions.csv')
    if not os.path.isfile(csv_file):
        print(f"File {csv_file} not found.")
        return solutions
    with open(csv_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                objective_str, path_str = line.split(';')
                objective = int(objective_str)
                path = [int(node.strip()) for node in path_str.strip().split(',')]
                solutions.append({'objective': objective, 'path': path})
    return solutions

def compute_similarity(solutions, measure='common_edges', comparison='to_best'):
    """Computes similarity measures for the solutions."""
    num_solutions = len(solutions)
    similarities = []
    objectives = []
    
    if comparison == 'to_best':
        # Find the best solution and exclude it
        best_solution = solutions[0]
        best_objective = best_solution['objective']
        best_path = best_solution['path']
        for sol in solutions:
            if sol['objective'] == best_objective:  # Skip the best solution
                continue
            sim = similarity_measure(sol['path'], best_path, measure)
            similarities.append(sim)
            objectives.append(sol['objective'])
    elif comparison == 'avg':
        # Compute average similarity to all other solutions
        for i in range(num_solutions):
            sol_i = solutions[i]
            sim_sum = 0
            count = 0
            for j in range(num_solutions):
                if i != j:  # Exclude self-comparison
                    sim = similarity_measure(sol_i['path'], solutions[j]['path'], measure)
                    sim_sum += sim
                    count += 1
            avg_sim = sim_sum / count if count > 0 else 0
            similarities.append(avg_sim)
            objectives.append(sol_i['objective'])
    else:
        print(f"Unknown comparison type: {comparison}")
        return None, None
    
    return objectives, similarities

def similarity_measure(path1, path2, measure='common_edges'):
    """Calculates similarity between two paths."""
    if measure == 'common_edges':
        edges1 = set()
        edges2 = set()
        for i in range(len(path1)):
            a = path1[i]
            b = path1[(i + 1) % len(path1)]
            edge = tuple(sorted((a, b)))  # Use sorted tuple to account for bidirectionality
            edges1.add(edge)
        for i in range(len(path2)):
            a = path2[i]
            b = path2[(i + 1) % len(path2)]
            edge = tuple(sorted((a, b)))
            edges2.add(edge)
        common_edges = edges1.intersection(edges2)
        similarity = len(common_edges) / len(edges1)
        return similarity * 100  # Percentage
    elif measure == 'common_nodes':
        nodes1 = set(path1)
        nodes2 = set(path2)
        common_nodes = nodes1.intersection(nodes2)
        similarity = len(common_nodes) / len(nodes1)
        return similarity * 100  # Percentage
    else:
        print(f"Unknown measure type: {measure}")
        return 0

def plot_similarity(objectives, similarities, instance_name, measure, comparison):
    """Plots the similarity chart and computes the correlation coefficient."""
    plt.figure(figsize=(10, 6))
    plt.scatter(objectives, similarities, alpha=0.6)
    plt.xlabel('Objective Function Value')
    plt.ylabel(f'Similarity (%) ({measure.replace("_", " ").title()})')
    plt.title(f'{instance_name}: Similarity ({comparison.replace("_", " ").title()})')
    
    # Calculate and display correlation coefficient
    corr_coef, _ = pearsonr(objectives, similarities)
    plt.annotate(f'Correlation Coefficient: {corr_coef:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=12, ha='right', va='top', bbox=dict(boxstyle='round', fc='w'))
    
    # Save the plot
    plot_filename = f'{instance_name}_{measure}_{comparison}.png'
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, plot_filename))
    plt.close()
    print(f"Plot saved: {plot_filename}")

def main():
    # Instances
    instances = ['TSPA', 'TSPB']  # Replace with your actual instance names
    
    # Similarity measures and comparisons
    measures = ['common_edges', 'common_nodes']
    comparisons = ['to_best', 'avg']
    
    for instance in instances:
        print(f"Processing instance: {instance}")
        solutions = read_solutions(instance)
        if not solutions:
            continue
        for measure in measures:
            for comparison in comparisons:
                objectives, similarities = compute_similarity(solutions, measure, comparison)
                if objectives is None or similarities is None:
                    continue
                plot_similarity(objectives, similarities, instance, measure, comparison)

if __name__ == "__main__":
    main()
