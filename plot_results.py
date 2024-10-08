import matplotlib.pyplot as plt
import numpy as np
import os

# Directories
instances_dir = 'inputs'
solutions_dir = 'outputs'
plots_dir = 'plots'

# Create the plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Function to load nodes
def load_nodes(instance):
    nodes_file = os.path.join(instances_dir, f'{instance}.csv')
    if not os.path.isfile(nodes_file):
        print(f"Nodes file {nodes_file} not found.")
        return None
    try:
        nodes = np.genfromtxt(nodes_file, delimiter=';')
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)  # Handle single-node case
        return nodes
    except Exception as e:
        print(f"Error loading {nodes_file}: {e}")
        return None

# Function to load solution path
def load_solution(method, instance):
    path_file = os.path.join(solutions_dir, instance, f'{method}.csv')
    if not os.path.isfile(path_file):
        print(f"Solution file {path_file} not found.")
        return None
    try:
        with open(path_file, 'r') as f:
            lines = f.read().splitlines()
            if not lines:
                print(f"Solution file {path_file} is empty.")
                return None
            # Extract path
            path = [int(line.strip()) for line in lines]
            return path
    except Exception as e:
        print(f"Error loading {path_file}: {e}")
        return None

# Validate path to ensure no revisits
def validate_path(path):
    path = path[:-1]  # Exclude the last node (same as the first)
    if len(path) != len(set(path)):
        print("Path has revisits!")
        return False
    else:
        print("Path is valid, no revisits.")
        return True

# Dynamically retrieve all instances from the inputs directory
instances = sorted([os.path.splitext(f)[0] for f in os.listdir(instances_dir) if f.lower().endswith('.csv')])
if not instances:
    print("No instances found in the 'inputs' directory.")
    exit()

print(f"Detected Instances: {instances}")

# Dynamically retrieve all methods by scanning the outputs directory
methods_set = set()
for instance in instances:
    instance_output_dir = os.path.join(solutions_dir, instance)
    if not os.path.isdir(instance_output_dir):
        print(f"Output directory for instance '{instance}' not found. Skipping.")
        continue
    for file in os.listdir(instance_output_dir):
        if file.endswith('.csv'):
            method_name = file[:-len('.csv')].strip()
            methods_set.add(method_name)

methods = sorted(methods_set)
if not methods:
    print("No methods found in the 'outputs' directory.")
    exit()

print(f"Detected Methods: {methods}")

# Iterate over each method to generate plots
for method in methods:
    # Prepare method-specific plot
    num_instances = len(instances)
    
    # Determine subplot grid size (e.g., 2x2, 3x3)
    cols = 2  # You can adjust this based on your preference
    rows = (num_instances + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))
    fig.suptitle(f'Best Solutions using {method}', fontsize=20)
    
    # Flatten axes array for easy iteration; handle cases where rows=1 or cols=1
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    for idx, instance in enumerate(instances):
        ax = axes[idx]
        nodes = load_nodes(instance)
        if nodes is None:
            ax.set_title(f"{instance} - Nodes file missing", fontsize=16)
            ax.axis('off')
            continue

        best_path = load_solution(method, instance)
        if best_path is None:
            ax.set_title(f"{instance} - Solution missing", fontsize=16)
            ax.axis('off')
            continue
        
        # Validate the best path before using it
        if not validate_path(best_path):
            ax.set_title(f"{instance} - Invalid path", fontsize=16)
            ax.axis('off')
            continue

        try:
            # Ensure that the path indices are within bounds
            if max(best_path) >= len(nodes) or min(best_path) < 0:
                raise IndexError("Node index out of bounds.")
            
            x_coords = nodes[best_path, 0]
            y_coords = nodes[best_path, 1]
            costs = nodes[best_path, 2]
        except IndexError as e:
            ax.set_title(f"{instance} - Invalid node indices", fontsize=16)
            ax.axis('off')
            continue

        # Plot the cycle by connecting the path back to the start
        ax.plot(
            np.append(x_coords, x_coords[0]),
            np.append(y_coords, y_coords[0]),
            'b-', lw=2, label="Cycle"
        )

        # Scatter plot of nodes with color representing cost
        scatter = ax.scatter(
            x_coords, y_coords,
            c=costs, cmap='viridis', s=150, edgecolor='black', label="Nodes"
        )

        # Annotate each node with its cost
        for i, cost in enumerate(costs):
            ax.text(
                x_coords[i], y_coords[i],
                f'{cost}', fontsize=9, ha='right', va='bottom'
            )

        # Highlight the start node
        ax.scatter(
            x_coords[0], y_coords[0],
            color='red', s=200, zorder=5, label="Start Node"
        )

        # Set plot titles and labels
        ax.set_title(f'{instance}', fontsize=16)
        ax.set_xlabel('X Coordinate', fontsize=14)
        ax.set_ylabel('Y Coordinate', fontsize=14)
        ax.legend(fontsize=12)

        # Add colorbar for node costs
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Node Cost', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.grid(True)

    # Turn off any unused subplots
    for idx in range(num_instances, rows * cols):
        axes[idx].axis('off')

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Sanitize method name for filename
    sanitized_method = method.replace(' ', '_')
    plot_filename = f'{sanitized_method}.png'
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot for method '{method}' saved to '{plot_path}'")

print("All plots generated successfully.")