# --- Step 1: Imports ---
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
import math
import random
import os

# --- Step 2: Maze Generation ---
def generate_random_maze(rows, cols):
    """
    Generates a random maze using a randomized Depth-First Search algorithm.
    The maze is guaranteed to be solvable. (Adapted from classical solver)
    """
    # Ensure dimensions are odd for a proper maze structure
    rows = rows if rows % 2 != 0 else rows + 1
    cols = cols if cols % 2 != 0 else cols + 1
    
    maze = [[ 'W' for _ in range(cols)] for _ in range(rows)]
    stack = [(0, 0)]
    maze[0][0] = '.'

    while stack:
        r, c = stack[-1]
        neighbors = []
        
        # Check for neighbors 2 cells away
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 'W':
                neighbors.append((nr, nc))
        
        if neighbors:
            nr, nc = random.choice(neighbors)
            maze[nr][nc] = '.'
            maze[r + (nr - r) // 2][c + (nc - c) // 2] = '.'
            stack.append((nr, nc))
        else:
            stack.pop()

    maze[0][0] = 'S'
    maze[rows - 1][cols - 1] = 'E'
    return maze

# --- Step 3: Classical Helper Functions ---
def find_start_end(layout):
    """Finds the start and end coordinates in the maze layout."""
    start, end = None, None
    for r, row_data in enumerate(layout):
        for c, cell in enumerate(row_data):
            if cell == 'S': start = (r, c)
            elif cell == 'E': end = (r, c)
    if start is None or end is None: raise ValueError("Maze must contain 'S' and 'E'.")
    return start, end

def is_valid_move(r, c, rows, cols):
    """Checks if a coordinate is within the maze bounds."""
    return 0 <= r < rows and 0 <= c < cols

def find_all_simple_paths(layout, start, end):
    """Finds all simple (no-cycle) paths from start to end using DFS."""
    rows, cols = len(layout), len(layout[0])
    paths, stack = [], [(start, [start])]
    while stack:
        (r, c), path = stack.pop()
        if (r, c) == end:
            paths.append(path)
            continue
        for dr, dc in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            nr, nc = r + dr, c + dc
            if is_valid_move(nr, nc, rows, cols) and (nr, nc) not in path:
                stack.append(((nr, nc), path + [(nr, nc)]))
    return paths

def is_path_valid(layout, path):
    """Checks if a given path hits any walls."""
    for r, c in path:
        if layout[r][c] == 'W': return False
    return True

# --- Step 4: Dynamic Oracle Builder ---
def create_oracle(n_qubits, winning_states):
    """Creates a Grover oracle for a list of winning binary states."""
    oracle_circuit = QuantumCircuit(n_qubits + 1, name="Oracle")
    for state in winning_states:
        for i, bit in enumerate(reversed(state)):
            if bit == '0': oracle_circuit.x(i)
        oracle_circuit.mcx(list(range(n_qubits)), n_qubits)
        for i, bit in enumerate(reversed(state)):
            if bit == '0': oracle_circuit.x(i)
        oracle_circuit.barrier()
    return oracle_circuit

# --- Step 5: Visualization ---
def visualize_maze_solution(layout, path_coords, title="Maze Solution"):
    """Draws the maze and highlights the solution path."""
    rows, cols = len(layout), len(layout[0])
    start_node, end_node = find_start_end(layout)
    G = nx.grid_2d_graph(rows, cols)

    for r in range(rows):
        for c in range(cols):
            if layout[r][c] == 'W' and G.has_node((c, r)):
                G.remove_node((c, r))
    
    pos = {(c, r): (c, -r) for r, row in enumerate(layout) for c, val in enumerate(row)}
    node_colors = ['green' if (r, c) == start_node else 'red' if (r, c) == end_node else 'lightblue' for c, r in G.nodes()]
    path_edges = [((p1[1], p1[0]), (p2[1], p2[0])) for p1, p2 in zip(path_coords, path_coords[1:])]
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=4)
    labels = { (c,r): f'({r},{c})' for c,r in G.nodes() }
    labels.update({(start_node[1], start_node[0]): 'S', (end_node[1], end_node[0]): 'E'})
    nx.draw_networkx_labels(G, pos, labels={k:v for k,v in labels.items() if k in G.nodes()}, font_color='black')
    plt.title(title)
    plt.show()

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Solve a dynamically generated maze using Grover's algorithm.")
    parser.add_argument('--rows', type=int, default=5, help='Number of rows for the random maze.')
    parser.add_argument('--cols', type=int, default=5, help='Number of columns for the random maze.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generator.')
    parser.add_argument('--visualize_grovers', action='store_true', help="Save a plot of amplitudes for each Grover iteration.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    print("--- Dynamic Quantum Maze Solver ---")
    
    maze_layout = generate_random_maze(args.rows, args.cols)
    start_node, end_node = find_start_end(maze_layout)
    all_paths = find_all_simple_paths(maze_layout, start_node, end_node)
    
    if not all_paths:
        print("No paths found from Start to End.")
        return
    
    num_paths = len(all_paths)
    n_qubits = (num_paths - 1).bit_length() if num_paths > 0 else 0
    search_space_size = 2**n_qubits
    
    print(f"Found {num_paths} potential paths. Using {n_qubits} qubits for a search space of {search_space_size}.")
    
    winning_states, path_map = [], {}
    for i, p in enumerate(all_paths):
        binary_rep = format(i, f'0{n_qubits}b')
        path_map[binary_rep] = p
        if is_path_valid(maze_layout, p):
            winning_states.append(binary_rep)

    if not winning_states:
        print("No valid paths found (all paths hit walls).")
        return
        
    print(f"Found {len(winning_states)} valid solution(s).")
    print(f"Winning binary state(s): {winning_states}")

    num_solutions = len(winning_states)
    iterations = round(math.pi / 4 * math.sqrt(search_space_size / num_solutions)) if num_solutions > 0 else 0
    print(f"Optimal number of Grover iterations: {iterations}")

    oracle = create_oracle(n_qubits, winning_states)
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))

    if args.visualize_grovers:
        def show_plot_from_data(sv_data, title):
            """Generates and displays a plot from statevector data."""
            # Use a dark theme for better visibility
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # This calculation correctly gets the probabilities for the data qubits
            # by summing the probabilities of states where the ancilla is |0> and |1>
            probs_dict = {format(i, f'0{n_qubits}b'): np.abs(sv_data[i])**2 + np.abs(sv_data[i + 2**n_qubits])**2 for i in range(2**n_qubits)}
            all_states = [format(i, f'0{n_qubits}b') for i in range(2**n_qubits)]
            
            # Since there can be thousands of states, we don't label the x-axis ticks
            ax.set_xticks([])
            
            probs = [probs_dict.get(s, 0) for s in all_states]
            
            # Plot all probabilities as a line graph
            ax.plot(all_states, probs, color='cyan', linewidth=1.5)
            ax.fill_between(all_states, probs, color='cyan', alpha=0.2)
            
            # Highlight the winning state(s) with a distinct marker
            winning_indices = [int(ws, 2) for ws in winning_states]
            winning_probs = [probs[i] for i in winning_indices]
            # Need to get the string representation for the scatter plot x-axis
            winning_labels = [all_states[i] for i in winning_indices]
            ax.scatter(winning_labels, winning_probs, color='magenta', s=50, zorder=10, label='Winning State')

            ax.set(xlabel=f'Computational Basis States ({search_space_size} total)', ylabel='Probability', title=title, ylim=(0,1))
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.legend()
            fig.tight_layout()
            
            # This line will pause the script and show the plot.
            # The script will only continue after you close the plot window.
            plt.show()
            
            # Reset style to default so it doesn't affect other plots
            plt.style.use('default')

        print("\n--- Displaying Grover's Algorithm Step-by-Step Plots (Optimized) ---")
        if n_qubits > 12:
            print(f"WARNING: Visualizing {search_space_size} states may be slow.")

        state_sim = AerSimulator(method='statevector')
        
        # Determine which steps to save for visualization (approx 10 plots)
        if iterations > 0:
            # Ensure we have at least 2 frames (start and end) and at most 10.
            num_frames = min(iterations + 1, 10) 
            # np.linspace gives evenly spaced steps. np.unique handles cases with few iterations.
            steps_to_save = np.unique(np.linspace(0, iterations, num=num_frames, dtype=int))
        else:
            steps_to_save = [0]
        
        # Build the full circuit with save points ONLY at the selected steps
        vis_circuit = QuantumCircuit(n_qubits + 1)
        vis_circuit.h(range(n_qubits))
        vis_circuit.x(n_qubits)
        vis_circuit.h(n_qubits)

        if 0 in steps_to_save:
            vis_circuit.save_statevector(label='step_0')

        for i in range(1, iterations + 1):
            vis_circuit.compose(oracle, inplace=True)
            vis_circuit.compose(diffuser, inplace=True, qubits=range(n_qubits))
            if i in steps_to_save:
                vis_circuit.save_statevector(label=f'step_{i}')
        
        print(f"Will generate visualizations for steps: {steps_to_save}")
        print("Running a single, consolidated simulation...")
        result = state_sim.run(transpile(vis_circuit, state_sim)).result()
        print("Simulation complete. Displaying plots from saved states...")

        # Now, loop through the results and generate plots
        for i in steps_to_save:
            label = f'step_{i}'
            sv_data = result.data()[label]
            title = "Initial State (Superposition)" if i == 0 else f"After Iteration {i}"
            print(f"Displaying plot for iteration {i}. Please close the window to continue...")
            show_plot_from_data(sv_data, title)
        
        print("\n--- Visualization Finished ---")

    # This part remains for the final, measured simulation
    circuit = QuantumCircuit(n_qubits + 1, n_qubits)
    circuit.h(range(n_qubits))
    circuit.x(n_qubits)
    circuit.h(n_qubits)
    circuit.barrier()

    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.compose(diffuser, inplace=True, qubits=range(n_qubits))
        circuit.barrier()

    circuit.measure(range(n_qubits), range(n_qubits))
    
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    counts = result.get_counts(compiled_circuit)

    print("\n--- Simulation Results ---")
    print("Counts:", counts)
    
    solution_binary = max(counts, key=counts.get)
    print(f"Quantum solver found solution: '{solution_binary}'")
    
    if solution_binary in path_map:
        solution_path = path_map[solution_binary]
        if solution_binary in winning_states:
            print("Solution is confirmed to be a valid path.")
            visualize_maze_solution(maze_layout, solution_path)
        else:
            print("Warning: Most probable state was not a valid path.")
            visualize_maze_solution(maze_layout, solution_path, title="Invalid Path Found by Solver")
    else:
        print(f"Error: Could not decode the solution '{solution_binary}'.")

if __name__ == "__main__":
    main()

