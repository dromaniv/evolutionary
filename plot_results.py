import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import os

# Directories
INPUTS_DIR = 'inputs'
OUTPUTS_DIR = 'outputs'
PLOTS_DIR = 'plots'
ANIMATIONS_DIR = 'animations'

# Create directories if they don't exist
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(ANIMATIONS_DIR, exist_ok=True)

def load_nodes(instance, max_nodes=100):
    nodes_file = os.path.join(INPUTS_DIR, f'{instance}.csv')
    if not os.path.isfile(nodes_file):
        print(f"Nodes file {nodes_file} not found.")
        return None
    try:
        nodes = np.genfromtxt(nodes_file, delimiter=';')
        if nodes.ndim == 1:
            nodes = nodes.reshape(1, -1)  # Handle single-node case
        nodes = nodes[:max_nodes]
        return nodes
    except Exception as e:
        print(f"Error loading {nodes_file}: {e}")
        return None

def load_solution(method, instance, max_nodes=100):
    path_file = os.path.join(OUTPUTS_DIR, instance, f'{method}.csv')
    if not os.path.isfile(path_file):
        print(f"Solution file {path_file} not found.")
        return None
    try:
        with open(path_file, 'r') as f:
            lines = f.read().splitlines()
            if not lines:
                print(f"Solution file {path_file} is empty.")
                return None
            path = [int(line.strip()) for line in lines if int(line.strip()) < max_nodes]
            if path[0] != path[-1]:
                path.append(path[0])
            return path
    except Exception as e:
        print(f"Error loading {path_file}: {e}")
        return None

def validate_path(path):
    path_without_return = path[:-1]  # Exclude the last node (same as the first)
    if len(path_without_return) != len(set(path_without_return)):
        print("Path has revisits!")
        return False
    return True

def plot_instance(ax, instance, method, nodes, best_path):
    if nodes is None or len(nodes) < 2:
        ax.set_title(f"{instance} - Insufficient nodes", fontsize=16)
        ax.axis('off')
        return

    if best_path is None or len(best_path) < 2:
        ax.set_title(f"{instance} - Solution missing or insufficient", fontsize=16)
        ax.axis('off')
        return

    if not validate_path(best_path):
        ax.set_title(f"{instance} - Invalid path", fontsize=16)
        ax.axis('off')
        return

    try:
        if max(best_path) >= len(nodes) or min(best_path) < 0:
            raise IndexError("Node index out of bounds.")

        x_coords = nodes[best_path, 0]
        y_coords = nodes[best_path, 1]
        costs = nodes[:, 2]
    except IndexError as e:
        ax.set_title(f"{instance} - Invalid node indices", fontsize=16)
        ax.axis('off')
        return

    # Plot the cycle
    ax.plot(
        x_coords,
        y_coords,
        'b-', lw=2, label="Cycle"
    )

    # Scatter plot of nodes with color representing cost
    scatter = ax.scatter(
        nodes[:, 0], nodes[:, 1],
        c=costs, cmap='plasma', s=100, edgecolor='black', label="Nodes", alpha=0.7
    )

    # Annotate each node with its cost
    for i, (x, y, cost) in enumerate(zip(nodes[:, 0], nodes[:, 1], costs)):
        ax.text(
            x, y,
            f'{int(cost)}', fontsize=10, ha='center', va='center', color='black', zorder=6
        )

    # Highlight the path nodes with a distinct color
    ax.scatter(
        x_coords, y_coords,
        color='lime', s=150, edgecolor='black', zorder=5, label="Path Nodes"
    )

    # Highlight the start node
    ax.scatter(
        x_coords[0], y_coords[0],
        color='red', s=200, edgecolor='black', zorder=7, label="Start Node"
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

def animate_instance(instance, method, nodes, best_path):
    if nodes is None or len(nodes) < 2:
        print(f"Nodes for instance '{instance}' not found or insufficient. Skipping animation.")
        return

    if best_path is None or len(best_path) < 2:
        print(f"Solution for method '{method}' and instance '{instance}' not found or insufficient. Skipping animation.")
        return

    if not validate_path(best_path):
        print(f"Invalid path for method '{method}' and instance '{instance}'. Skipping animation.")
        return

    try:
        if max(best_path) >= len(nodes) or min(best_path) < 0:
            raise IndexError("Node index out of bounds.")

        x_coords = nodes[best_path, 0]
        y_coords = nodes[best_path, 1]
        costs = nodes[:, 2]
    except IndexError as e:
        print(f"Error with node indices for method '{method}' and instance '{instance}'. Skipping animation.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize plot elements
    line, = ax.plot([], [], 'g--', lw=2, label="Current Path")
    complete_line, = ax.plot([], [], 'b-', lw=2, label="Completed Path")
    scatter = ax.scatter(
        nodes[:, 0], nodes[:, 1],
        c=costs, cmap='plasma', s=100, edgecolor='black', label="Nodes", alpha=0.7
    )

    # Highlight the start node
    start_node_scatter = ax.scatter(
        x_coords[0], y_coords[0],
        color='red', s=200, edgecolor='black', zorder=5, label="Start Node"
    )

    # Set plot titles and labels
    ax.set_title(f'{instance} - {method}', fontsize=16)
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Node Cost', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.grid(True)

    # Set axis limits with padding
    padding = 0.05
    x_min, x_max = min(nodes[:, 0]), max(nodes[:, 0])
    y_min, y_max = min(nodes[:, 1]), max(nodes[:, 1])
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)
    ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)

    # Function to initialize the plot
    def init():
        line.set_data([], [])
        complete_line.set_data([], [])
        return line, complete_line

    # Function to update the plot
    def animate(i):
        if i < len(x_coords):
            x = x_coords[:i+1]
            y = y_coords[:i+1]
            line.set_data(x, y)
            complete_line.set_data([], [])
        else:
            line.set_data([], [])
            complete_line.set_data(x_coords, y_coords)
        return line, complete_line

    num_frames = len(x_coords) + 20  # Extra frames for pause at the end

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=num_frames, interval=100, blit=True
    )

    # Save animation to file
    sanitized_method = method.replace(' ', '_')
    sanitized_instance = instance.replace(' ', '_')
    animation_filename = f'{sanitized_instance}/{sanitized_method}.gif'
    animation_path = os.path.join(ANIMATIONS_DIR, animation_filename)

    try:
        ani.save(animation_path, writer=PillowWriter(fps=10))
        print(f"Animation for method '{method}' and instance '{instance}' saved to '{animation_path}'")
    except Exception as e:
        print(f"Failed to save animation for method '{method}' and instance '{instance}': {e}")

    plt.close()

def main():
    # Get instances
    instances = sorted([os.path.splitext(f)[0] for f in os.listdir(INPUTS_DIR) if f.lower().endswith('.csv')])
    if not instances:
        print("No instances found in the 'inputs' directory.")
        return

    print(f"Detected Instances: {instances}")

    for instance in instances:
        os.makedirs(os.path.join(ANIMATIONS_DIR, instance), exist_ok=True)
        

    # Get methods
    methods_set = set()
    for instance in instances:
        instance_output_dir = os.path.join(OUTPUTS_DIR, instance)
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
        return

    print(f"Detected Methods: {methods}")

    # Generate plots for each method
    for method in methods:
        num_instances = len(instances)
        cols = 2  # Adjust as needed
        rows = (num_instances + cols - 1) // cols  # Ceiling division

        fig, axes = plt.subplots(rows, cols, figsize=(10 * cols, 10 * rows))
        fig.suptitle(f'Best Solutions using {method}', fontsize=20)

        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, instance in enumerate(instances):
            ax = axes[idx]
            nodes = load_nodes(instance)
            best_path = load_solution(method, instance)
            plot_instance(ax, instance, method, nodes, best_path)

        # Turn off any unused subplots
        for idx in range(num_instances, rows * cols):
            axes[idx].axis('off')

        # Adjust layout and save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        sanitized_method = method.replace(' ', '_')
        plot_filename = f'{sanitized_method}.png'
        plot_path = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plot for method '{method}' saved to '{plot_path}'")

    # Generate animations
    for method in methods:
        for instance in instances:
            nodes = load_nodes(instance)
            best_path = load_solution(method, instance)
            animate_instance(instance, method, nodes, best_path)

if __name__ == "__main__":
    main()
