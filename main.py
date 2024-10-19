import csv
import math
import random
from math import ceil
import numpy as np

def read_csv(file_path):
    """
    Reads the CSV file and returns a list of nodes.
    Each node is a tuple: (x, y, cost)
    """
    nodes = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        for row in reader:
            if len(row) != 3:
                continue  # Skip invalid rows
            try:
                x = int(row[0])
                y = int(row[1])
                cost = int(row[2])
                nodes.append( (x, y, cost) )
            except ValueError:
                continue  # Skip rows with invalid integers
    return nodes

def compute_distance_matrix(nodes):
    """
    Computes the Euclidean distance matrix between nodes, rounded to nearest integer.
    Returns a numpy array of shape (n, n)
    """
    n = len(nodes)
    distance_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        xi, yi, _ = nodes[i]
        for j in range(i+1, n):
            xj, yj, _ = nodes[j]
            dist = math.sqrt( (xi - xj)**2 + (yi - yj)**2 )
            dist = int(round(dist))
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
    return distance_matrix

def compute_objective(path, distance_matrix, nodes):
    """
    Computes the objective function: sum of path lengths + sum of node costs
    """
    total_distance = 0
    k = len(path)
    for i in range(k):
        total_distance += distance_matrix[path[i]][path[(i+1)%k]]
    total_cost = sum([nodes[node][2] for node in path])
    return total_distance + total_cost

def random_solution(start_node, nodes, distance_matrix, k):
    """
    Generates a random solution: selects k nodes randomly and arranges them in a random cycle.
    Returns the path and its objective value.
    """
    selected_nodes = random.sample(range(len(nodes)), k)
    random.shuffle(selected_nodes)
    objective = compute_objective(selected_nodes, distance_matrix, nodes)
    return selected_nodes, objective

def nearest_neighbor_end(start_node, nodes, distance_matrix, k):
    """
    Nearest Neighbor heuristic: always add the nearest unselected node to the end of the path.
    Returns the path and its objective value.
    """
    path = [start_node]
    selected = set(path)
    while len(path) < k:
        last = path[-1]
        # Find the nearest unselected node
        nearest = None
        min_dist = float('inf')
        for node in range(len(nodes)):
            if node not in selected:
                dist = distance_matrix[last][node]
                if dist < min_dist:
                    min_dist = dist
                    nearest = node
        if nearest is not None:
            path.append(nearest)
            selected.add(nearest)
        else:
            break  # No more nodes to add
    objective = compute_objective(path, distance_matrix, nodes)
    return path, objective

def nearest_neighbor_any(start_node, nodes, distance_matrix, k):
    """
    Nearest Neighbor heuristic: insert the nearest unselected node at the position that minimizes the path length.
    Returns the path and its objective value.
    """
    path = [start_node]
    selected = set(path)
    while len(path) < k:
        # Find the nearest unselected node to any node in the path
        nearest = None
        min_dist = float('inf')
        for node in range(len(nodes)):
            if node not in selected:
                for p_node in path:
                    dist = distance_matrix[p_node][node]
                    if dist < min_dist:
                        min_dist = dist
                        nearest = node
        if nearest is not None:
            # Find the best position to insert the nearest node
            best_increase = float('inf')
            best_pos = None
            for i in range(len(path)):
                j = (i + 1) % len(path)
                increase = distance_matrix[path[i]][nearest] + distance_matrix[nearest][path[j]] - distance_matrix[path[i]][path[j]]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = j
            if best_pos is not None:
                path.insert(best_pos, nearest)
                selected.add(nearest)
            else:
                # If no best position found, append at the end
                path.append(nearest)
                selected.add(nearest)
        else:
            break  # No more nodes to add
    objective = compute_objective(path, distance_matrix, nodes)
    return path, objective

def greedy_cycle(start_node, nodes, distance_matrix, k):
    """
    Greedy Cycle heuristic: combines distance and node cost.
    Selects the next node that minimizes (distance + cost).
    Returns the path and its objective value.
    """
    path = [start_node]
    selected = set(path)
    while len(path) < k:
        best_node = None
        best_score = float('inf')
        for node in range(len(nodes)):
            if node not in selected:
                # Compute the best insertion position
                min_increase = float('inf')
                for i in range(len(path)):
                    j = (i + 1) % len(path)
                    increase = distance_matrix[path[i]][node] + distance_matrix[node][path[j]] - distance_matrix[path[i]][path[j]]
                    if increase < min_increase:
                        min_increase = increase
                # Define score as increase + node cost
                score = min_increase + nodes[node][2]
                if score < best_score:
                    best_score = score
                    best_node = node
        if best_node is not None:
            # Insert the best_node at the position that minimizes the increase
            best_increase = float('inf')
            best_pos = None
            for i in range(len(path)):
                j = (i + 1) % len(path)
                increase = distance_matrix[path[i]][best_node] + distance_matrix[best_node][path[j]] - distance_matrix[path[i]][path[j]]
                if increase < best_increase:
                    best_increase = increase
                    best_pos = j
            if best_pos is not None:
                path.insert(best_pos, best_node)
                selected.add(best_node)
            else:
                # Append at the end if no better position found
                path.append(best_node)
                selected.add(best_node)
        else:
            break  # No more nodes to add
    objective = compute_objective(path, distance_matrix, nodes)
    return path, objective

def greedy_2_regret(start_node, nodes, distance_matrix, k):
    """
    Greedy 2-regret heuristic: Select the node that causes the largest regret 
    (the difference between the best and second-best insertion cost + node cost).
    Returns the path and its objective value.
    """
    path = [start_node]
    selected = set(path)
    
    while len(path) < k:
        best_node = None
        best_regret = -float('inf')
        for node in range(len(nodes)):
            if node not in selected:
                # Compute the best and second-best insertion costs, including node cost
                insertion_costs = []
                for i in range(len(path)):
                    j = (i + 1) % len(path)
                    # Insertion cost: distance increase + node cost
                    increase = distance_matrix[path[i]][node] + distance_matrix[node][path[j]] - distance_matrix[path[i]][path[j]]
                    insertion_cost = increase + nodes[node][2]
                    insertion_costs.append(insertion_cost)
                if not insertion_costs:
                    continue
                insertion_costs.sort()
                best_increase = insertion_costs[0]
                second_best_increase = insertion_costs[1] if len(insertion_costs) > 1 else insertion_costs[0]
                regret = second_best_increase - best_increase
                if regret > best_regret:
                    best_regret = regret
                    best_node = node
        
        if best_node is not None:
            # Insert the best_node at the position that minimizes the insertion cost
            best_increase = float('inf')
            best_pos = None
            for i in range(len(path)):
                j = (i + 1) % len(path)
                increase = distance_matrix[path[i]][best_node] + distance_matrix[best_node][path[j]] - distance_matrix[path[i]][path[j]]
                insertion_cost = increase + nodes[best_node][2]
                if insertion_cost < best_increase:
                    best_increase = insertion_cost
                    best_pos = j
            if best_pos is not None:
                path.insert(best_pos, best_node)
                selected.add(best_node)
            else:
                # Append at the end if no better position found
                path.append(best_node)
                selected.add(best_node)
        else:
            break  # No more nodes to add
    
    objective = compute_objective(path, distance_matrix, nodes)
    return path, objective

def greedy_2_regret_weighted(start_node, nodes, distance_matrix, k, weight_regret=0.5, weight_change=0.5):
    """
    Greedy 2-regret with weighted sum criterion: Selects the node that maximizes 
    the weighted sum of 2-regret and the best objective change (distance + cost).
    Returns the path and its objective value.
    """
    path = [start_node]
    selected = set(path)
    
    while len(path) < k:
        best_node = None
        best_score = -float('inf')
        for node in range(len(nodes)):
            if node not in selected:
                # Compute the best and second-best insertion costs, including node cost
                insertion_costs = []
                for i in range(len(path)):
                    j = (i + 1) % len(path)
                    # Insertion cost: distance increase + node cost
                    increase = distance_matrix[path[i]][node] + distance_matrix[node][path[j]] - distance_matrix[path[i]][path[j]]
                    insertion_cost = increase + nodes[node][2]
                    insertion_costs.append(insertion_cost)
                if not insertion_costs:
                    continue
                insertion_costs.sort()
                best_increase = insertion_costs[0]
                second_best_increase = insertion_costs[1] if len(insertion_costs) > 1 else insertion_costs[0]
                regret = second_best_increase - best_increase
                
                # Compute the score: weighted sum of regret and negative best_increase
                # Negative because lower insertion cost is better
                score = weight_regret * regret + weight_change * (-best_increase)
                if score > best_score:
                    best_score = score
                    best_node = node
        
        if best_node is not None:
            # Insert the best_node at the position that minimizes the insertion cost
            best_increase = float('inf')
            best_pos = None
            for i in range(len(path)):
                j = (i + 1) % len(path)
                increase = distance_matrix[path[i]][best_node] + distance_matrix[best_node][path[j]] - distance_matrix[path[i]][path[j]]
                insertion_cost = increase + nodes[best_node][2]
                if insertion_cost < best_increase:
                    best_increase = insertion_cost
                    best_pos = j
            if best_pos is not None:
                path.insert(best_pos, best_node)
                selected.add(best_node)
            else:
                # Append at the end if no better position found
                path.append(best_node)
                selected.add(best_node)
        else:
            break  # No more nodes to add
    
    objective = compute_objective(path, distance_matrix, nodes)
    return path, objective

def generate_solutions(nodes, distance_matrix, k, method_func, num_solutions=200, start_nodes=None, **kwargs):
    """
    Generates solutions using the specified method function.
    Returns a list of tuples: (path, objective)
    """
    solutions = []
    if start_nodes is not None:
        for start_node in start_nodes:
            if method_func.__name__ in ["random_solution"]:
                path, obj = method_func(None, nodes, distance_matrix, k)
            elif method_func.__name__ in ["greedy_2_regret_weighted"]:
                path, obj = method_func(start_node, nodes, distance_matrix, k, **kwargs)
            else:
                path, obj = method_func(start_node, nodes, distance_matrix, k)
            solutions.append( (path, obj) )
    else:
        for _ in range(num_solutions):
            path, obj = method_func(None, nodes, distance_matrix, k)
            solutions.append( (path, obj) )
    return solutions

def compute_statistics(solutions):
    """
    Given a list of solutions (path, objective), computes min, max, average of objective.
    Also finds the best solution (with minimum objective).
    Returns (min_obj, max_obj, avg_obj, best_path)
    """
    if not solutions:
        return None, None, None, None
    objectives = [obj for (_, obj) in solutions]
    min_obj = min(objectives)
    max_obj = max(objectives)
    avg_obj = sum(objectives) / len(objectives)
    # Find the best path
    best_index = objectives.index(min_obj)
    best_path = solutions[best_index][0]
    return min_obj, max_obj, avg_obj, best_path

def main():
    file_path = "inputs/TSPB.csv"
    # Read nodes from CSV
    nodes = read_csv(file_path)
    n = len(nodes)
    if n == 0:
        print("No valid nodes found in the CSV file.")
        return
    k = ceil(n / 2)
    print(f"Total nodes: {n}, Selecting k={k} nodes.")

    # Check if the number of nodes is at least 200
    if n < 200:
        print(f"Warning: The number of nodes ({n}) is less than 200. Adjusting num_solutions to {n}.")
        num_solutions = n
    else:
        num_solutions = 200

    # Define the list of start nodes (iterate through the first 200 nodes)
    start_nodes = list(range(min(num_solutions, n)))

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(nodes)
    print("Distance matrix computed.")

    # Define methods
    methods = [
        {"name": "Random Solution", "function": random_solution, "start_nodes": None, "num_solutions": num_solutions},
        {"name": "Nearest Neighbor (End Insertion)", "function": nearest_neighbor_end, "start_nodes": start_nodes, "num_solutions": None},
        {"name": "Nearest Neighbor (Any Insertion)", "function": nearest_neighbor_any, "start_nodes": start_nodes, "num_solutions": None},
        {"name": "Greedy Cycle", "function": greedy_cycle, "start_nodes": start_nodes, "num_solutions": None},
        {"name": "Greedy 2-Regret", "function": greedy_2_regret, "start_nodes": start_nodes, "num_solutions": None},
        {"name": "Greedy 2-Regret with Weighted Sum", 
         "function": greedy_2_regret_weighted, 
         "start_nodes": start_nodes, 
         "num_solutions": None, 
         "weights": {'weight_regret': 1, 'weight_change': 1}},
    ]

    # Generate solutions
    solutions = {}
    for method in methods:
        print(f"Generating {method['name']} solutions...")
        if method["name"] == "Greedy 2-Regret with Weighted Sum":
            sols = generate_solutions(
                nodes,
                distance_matrix,
                k,
                method['function'],
                num_solutions=method.get('num_solutions'),
                start_nodes=method.get('start_nodes'),
                weight_regret=method.get('weights', {}).get('weight_regret', 0.5),
                weight_change=method.get('weights', {}).get('weight_change', 0.5)
            )
        else:
            sols = generate_solutions(
                nodes,
                distance_matrix,
                k,
                method['function'],
                num_solutions=method.get('num_solutions'),
                start_nodes=method.get('start_nodes')
            )
        solutions[method['name']] = sols

    # Compute statistics
    stats = {}
    for method in methods:
        print(f"Computing statistics for {method['name']} solutions...")
        min_obj, max_obj, avg_obj, best_path = compute_statistics(solutions[method['name']])
        stats[method['name']] = (min_obj, max_obj, avg_obj, best_path)

    # Output results
    print("\n--- Computational Experiment Results ---")
    for method in methods:
        min_obj, max_obj, avg_obj, best_path = stats[method['name']]
        print(f"Method: {method['name']}")
        print(f"Min Objective: {min_obj}")
        print(f"Max Objective: {max_obj}")
        print(f"Average Objective: {avg_obj:.2f}")
        print(f"Best Solution: {best_path}\n")

if __name__ == "__main__":
    main()