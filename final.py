# --- Step 1: Imports ---
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import math
import random
import time
import heapq

# --- Step 2: Maze Generation ---
def generate_random_maze(rows, cols):
    """
    Generates a random 'perfect' maze (one unique path) using
    a randomized Depth-First Search algorithm.
    """
    rows = rows if rows % 2 != 0 else rows + 1
    cols = cols if cols % 2 != 0 else cols + 1
    
    maze = [[ 'W' for _ in range(cols)] for _ in range(rows)]
    stack = [(0, 0)]
    maze[0][0] = '.'

    while stack:
        r, c = stack[-1]
        neighbors = []
        
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

# --- [NEW FUNCTION] ---
def add_loops_to_maze(maze, num_loops=1):
    """
    Modifies a 'perfect' maze by knocking down internal walls to create loops.
    This results in multiple valid solution paths.
    """
    rows, cols = len(maze), len(maze[0])
    walls_removed = 0
    
    # Try a few times to find walls, in case we pick non-walls
    for _ in range(num_loops * 10): 
        if walls_removed >= num_loops:
            break
            
        # Pick a random internal cell (not the border)
        r = random.randint(1, rows - 2)
        c = random.randint(1, cols - 2)
        
        if maze[r][c] == 'W':
            # Check if it's a horizontal wall between two paths
            is_horizontal_divider = (maze[r-1][c] != 'W' and maze[r+1][c] != 'W')
            # Check if it's a vertical wall between two paths
            is_vertical_divider = (maze[r][c-1] != 'W' and maze[r][c+1] != 'W')
            
            # Ensure it's not already a path and it's a valid divider
            if (is_horizontal_divider and not is_vertical_divider) or \
               (is_vertical_divider and not is_horizontal_divider):
                maze[r][c] = '.'
                walls_removed += 1
                
    print(f"[Maze Gen] Created {walls_removed} extra path(s) by removing walls.")
    return maze

# --- Step 3: Classical Solver Components ---

def create_maze_graph(maze):
    """Converts a 2D list-based maze into a NetworkX graph."""
    rows, cols = len(maze), len(maze[0])
    graph = nx.Graph()
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] != 'W':
                graph.add_node((r, c))
                # Check neighbors
                for dr, dc in [(0, 1), (1, 0)]: # Only need to check right and down
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] != 'W':
                        graph.add_edge((r, c), (nr, nc))
    return graph

def manhattan_distance(a, b):
    """Heuristic function for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def solve_maze_astar(graph, start, end):
    """Solves the maze using A* search, returning path and visited nodes."""
    open_set = [(manhattan_distance(start, end), start)]
    predecessors = {start: None}
    g_score = {node: float('inf') for node in graph.nodes()}
    g_score[start] = 0
    visited_order = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited_order:
            continue
        visited_order.append(current)

        if current == end:
            path = []
            curr = end
            while curr is not None:
                path.append(curr)
                curr = predecessors.get(curr)
            return path[::-1], visited_order

        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float('inf')):
                predecessors[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + manhattan_distance(neighbor, end)
                heapq.heappush(open_set, (f_score, neighbor))
    return None, visited_order

# --- Step 4: Quantum Solver Helper Functions ---

def find_start_end(layout):
    """Finds the start and end coordinates in the maze layout."""
    start, end = None, None
    for r, row_data in enumerate(layout):
        for c, cell in enumerate(row_data):
            if cell == 'S': start = (r, c)
            elif cell == 'E': end = (r, c)
    return start, end

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
            # Check bounds AND not in path
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in path:
                stack.append(((nr, nc), path + [(nr, nc)]))
    return paths

def is_path_valid(layout, path):
    """Checks if a given path hits any walls."""
    return not any(layout[r][c] == 'W' for r, c in path)

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
    G = create_maze_graph(layout)
    
    pos = {(r, c): (c, -r) for r,c in G.nodes()}
    node_colors = ['green' if (r, c) == start_node else 'red' if (r, c) == end_node else 'lightblue' for r, c in G.nodes()]
    path_edges = list(zip(path_coords, path_coords[1:]))
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='orange', width=4)
    labels = { (r,c): f'({r},{c})' for r,c in G.nodes() }
    labels[start_node] = 'S'
    labels[end_node] = 'E'
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='black')
    plt.title(title)
    plt.show()

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Solve a maze using classical and quantum algorithms.")
    parser.add_argument('--rows', type=int, default=5, help='Rows for the random maze.')
    parser.add_argument('--cols', type=int, default=5, help='Columns for the random maze.')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generator.')
    parser.add_argument('--no-viz', action='store_true', help="Don't display the maze visualization.")
    
    # --- [NEW ARGUMENT] ---
    parser.add_argument('--loops', type=int, default=0, help='Number of extra walls to remove to create multiple paths.')
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # --- Generate Maze ---
    maze_layout = generate_random_maze(args.rows, args.cols)
    
    # --- [NEW LOGIC] ---
    if args.loops > 0:
        maze_layout = add_loops_to_maze(maze_layout, args.loops)
    
    start_node, end_node = find_start_end(maze_layout)
    
    # --- Quantum Solver ---
    print("\n--- Running Quantum Solver ---")
    q_start_time = time.perf_counter()
    
    # This is the classical bottleneck
    prep_start_time = time.perf_counter()
    all_paths = find_all_simple_paths(maze_layout, start_node, end_node)
    
    if not all_paths:
        print("No paths found. Halting.")
        return
        
    num_paths = len(all_paths)
    n_qubits = (num_paths - 1).bit_length() if num_paths > 0 else 0
    search_space_size = 2**n_qubits
    
    path_map, winning_states = {}, []
    for i, p in enumerate(all_paths):
        binary_rep = format(i, f'0{n_qubits}b')
        path_map[binary_rep] = p
        if is_path_valid(maze_layout, p):
            winning_states.append(binary_rep)
            
    prep_end_time = time.perf_counter()
    classical_prep_time = prep_end_time - prep_start_time

    if not winning_states:
        print("No valid paths found. Halting.")
        return
        
    num_solutions = len(winning_states)
    iterations = round(math.pi / 4 * math.sqrt(search_space_size / num_solutions)) if num_solutions > 0 else 0

    # Build Quantum Circuit
    sim_start_time = time.perf_counter()
    circuit = QuantumCircuit(n_qubits + 1, n_qubits)
    circuit.h(range(n_qubits)); circuit.x(n_qubits); circuit.h(n_qubits); circuit.barrier()
    
    oracle = create_oracle(n_qubits, winning_states)
    diffuser = QuantumCircuit(n_qubits, name="Diffuser")
    diffuser.h(range(n_qubits)); diffuser.x(range(n_qubits)); diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1); diffuser.x(range(n_qubits)); diffuser.h(range(n_qubits))

    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.compose(diffuser, inplace=True, qubits=range(n_qubits))
        circuit.barrier()
    circuit.measure(range(n_qubits), range(n_qubits))
    
    # Run Simulation
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    result = simulator.run(compiled_circuit, shots=1024).result()
    counts = result.get_counts(compiled_circuit)
    
    sim_end_time = time.perf_counter()
    quantum_sim_time = sim_end_time - sim_start_time
    
    solution_binary = max(counts, key=counts.get)
    q_solution_path = path_map.get(solution_binary)
    q_end_time = time.perf_counter() # Total time

    # --- Classical Solver ---
    print("\n--- Running Classical (A*) Solver ---")
    c_start_time = time.perf_counter()
    maze_graph = create_maze_graph(maze_layout)
    c_solution_path, c_visited_nodes = solve_maze_astar(maze_graph, start_node, end_node)
    c_end_time = time.perf_counter()

    # --- Final Comparison ---
    print("\n--- FINAL RESULTS ---")
    print(f"Maze Dimensions: {len(maze_layout)}x{len(maze_layout[0])}")
    print("-" * 25)
    print("Classical (A*) Solver:")
    print(f"  - Time Taken: {(c_end_time - c_start_time)*1000:.4f} ms")
    print(f"  - Path Found: {'Yes' if c_solution_path else 'No'}")
    print(f"  - Nodes Explored: {len(c_visited_nodes) if c_visited_nodes else 'N/A'}")
    print(f"  - Path Length: {len(c_solution_path) - 1 if c_solution_path else 'N/A'}")
    print("-" * 25)
    print("Quantum (Grover's) Solver:")
    print(f"  - Total Time: {(q_end_time - q_start_time)*1000:.4f} ms")
    print(f"      - Classical Prep Time: {classical_prep_time*1000:.4f} ms") # Bottleneck
    print(f"      - Quantum Sim Time:  {quantum_sim_time*1000:.4f} ms")
    print(f"  - Search Space Size (N): {search_space_size} (encoded in {n_qubits} qubits)")
    print(f"  - Number of Solutions (M): {num_solutions}")
    print(f"  - Oracle Queries (Iterations): {iterations} (sqrt(N/M))")
    print(f"  - Success Probability: {counts.get(solution_binary, 0)/1024:.2%}")
    q_path_is_valid = q_solution_path and solution_binary in winning_states
    print(f"  - Path Found: {'Yes' if q_path_is_valid else 'No'}")
    if q_path_is_valid:
        print(f"  - Path Length: {len(q_solution_path) - 1}")
    print("-" * 25)

    if not args.no_viz and q_path_is_valid:
        visualize_maze_solution(maze_layout, q_solution_path, title=f"Quantum Solver Result (Path Length: {len(q_solution_path)-1})")
        
    if not args.no_viz and c_solution_path:
        visualize_maze_solution(maze_layout, c_solution_path, title=f"Classical A* Result (Path Length: {len(c_solution_path)-1})")


if __name__ == "__main__":
    main()