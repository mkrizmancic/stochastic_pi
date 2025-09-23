import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy


class Node:
    """
    Represents a single node in the network, holding its state and performing local computations.
    """
    def __init__(self, node_id, num_nodes, epsilon_bar):
        rng = np.random.default_rng(42 + node_id)  # Seeded RNG for reproducibility
        # --- Persistent State Variables ---
        self.id = node_id
        # Estimate of the i-th component of the Fiedler vector
        self.x_i = rng.standard_normal()
        # Smoothed estimate of the second largest eigenvalue of W_bar
        self.y = rng.random()
        # Estimate of algebraic connectivity (lambda_2)
        self.z = rng.random()

        # --- System Parameters ---
        self.N = num_nodes
        self.epsilon_bar = epsilon_bar

        # --- Per-iteration temporary values ---
        self.b_i = 0.0
        self.b2_i = 0.0

    def compute_b_vectors(self, m_k, epsilon_k, neighbor_states, graph):
        """
        Computes the local vectors b_i[k] and b_{2,i}[k] for the current iteration k.
        This corresponds to Step 2 in Table II, using equations (19) and (20).
        """
        # Get x_j for all neighbors j
        sum_term = sum(neighbor_states[j] - self.x_i for j in graph.neighbors(self.id))

        # Equation (19)
        self.b_i = self.x_i + self.epsilon_bar * sum_term - m_k

        # Equation (20)
        self.b2_i = self.x_i + epsilon_k * sum_term - m_k

    def update_eigenvalue_estimate(self, y0_k, alpha_k):
        """
        Updates the local estimate y[k+1] and z[k+1].
        Corresponds to Steps 4 & 5 in Table II, using equations (11) and (12).
        """
        # Equation (11): Update smoothed eigenvalue estimate of W_bar
        self.y = self.y + alpha_k * (y0_k - self.y)

        # Equation (12): Update estimate of algebraic connectivity
        self.z = (1 - self.y) / self.epsilon_bar

    def update_eigenvector_estimate(self, norm_b2_k):
        """
        Performs the power iteration step to update x_i[k+1].
        Corresponds to Step 6 in Table II, using equation (22).
        """
        if norm_b2_k > 1e-9: # Avoid division by zero
            self.x_i = self.b2_i / norm_b2_k


class Network:
    """
    Manages the network of nodes, the graph topology, and the simulation flow.
    """
    def __init__(self, num_nodes, ideal_graph, connection_prob, params):
        self.N = num_nodes
        self.ideal_graph = ideal_graph
        self.p_c = connection_prob
        self.params = params

        self.nodes = {i: Node(i, self.N, params['epsilon_bar']) for i in range(self.N)}

        # Calculate the theoretical value for comparison
        self.theoretical_lambda2 = self._calculate_theoretical_lambda2()
        print(f"Theoretical Algebraic Connectivity of Expected Graph: {self.theoretical_lambda2:.4f}")

    def _calculate_theoretical_lambda2(self):
        """Calculates the true algebraic connectivity of the expected graph."""
        # The expected Laplacian has the same structure as the ideal one,
        # but with edge weights scaled by the connection probability p_c.
        L_bar = nx.laplacian_matrix(self.ideal_graph).toarray() * self.p_c
        eigenvalues = np.sort(np.linalg.eigvalsh(L_bar))
        return eigenvalues[1] # The second smallest eigenvalue

    def _create_random_graph_instance(self):
        """Creates a random instance of the graph for one time step k."""
        G_k = nx.Graph()
        G_k.add_nodes_from(self.ideal_graph)
        for u, v in self.ideal_graph.edges():
            if np.random.rand() < self.p_c:
                G_k.add_edge(u, v)
        return G_k

    def _run_consensus(self, initial_values, graph, num_steps, epsilon=0.1):
        """
        Simulates a simple average consensus protocol for a fixed number of steps.
        Each node repeatedly averages its value with its neighbors.
        """
        node_values = copy.deepcopy(initial_values)
        for _ in range(num_steps):
            new_values = {}
            for i in range(self.N):
                neighbor_vals = [node_values[j] for j in graph.neighbors(i)]
                # Average with self and neighbors
                new_values[i] = node_values[i] + epsilon * sum(n_val - node_values[i] for n_val in neighbor_vals)
            node_values = new_values
        # After enough iterations, all values are close to the true average
        return node_values

    def run_simulation(self, num_iterations, num_consensus_steps):
        """
        Runs the main distributed estimation algorithm from Table II.
        """
        history = []

        for node in self.nodes.values():
            print(f"Node {node.id}: Initial x_i = {node.x_i:.4f}, y = {node.y:.4f}, z = {node.z:.4f}")
        print("Average: ", np.mean([node.x_i for node in self.nodes.values()]))
        print()

        for k in range(num_iterations):
            # --- Get diminishing step sizes for iteration k --- CHECKED
            alpha_k = self.params['alpha0'] / ((k + 1) ** self.params['beta'])
            epsilon_k = self.params['epsilon0'] / ((k + 1) ** self.params['gamma'])

            # --- Get the random graph instance for this iteration --- CHECKED
            # G_k = self._create_random_graph_instance()
            G_k = self.ideal_graph  # Using the ideal graph for simplicity in this implementation

            # --- Get current Fiedler vector estimates from all nodes --- CHECKED
            current_x = {i: node.x_i for i, node in self.nodes.items()}

            # --- Step 1: Run consensus to get m[k] = mean(x[k]) --- CHECKED
            m_k = self._run_consensus(current_x, G_k, num_consensus_steps, self.params['epsilon_bar'])
            print(f"m[{k}] = {m_k}")

            # --- Step 2: Each node computes its b_i and b2_i vectors locally --- CHECKED
            for i, node in self.nodes.items():
                node.compute_b_vectors(m_k[i], epsilon_k, current_x, G_k)
                print(f"Node {i}: b_i = {node.b_i:.4f}, b2_i = {node.b2_i:.4f}")

            # --- Step 3: Run consensus for Rayleigh Ratio and Norm --- CHECKED
            # Initial values for the two parallel consensus rounds
            rayleigh_num_vals = {i: node.x_i * node.b_i for i, node in self.nodes.items()}
            rayleigh_den_vals = {i: node.x_i**2 for i, node in self.nodes.items()}
            norm_sq_vals = {i: node.b2_i**2 for i, node in self.nodes.items()}

            # The paper combines these, but simulating them separately is clearer --- CHECKED
            sum_rayleigh_num = self._run_consensus(rayleigh_num_vals, G_k, num_consensus_steps, self.params['epsilon_bar'])
            sum_rayleigh_den = self._run_consensus(rayleigh_den_vals, G_k, num_consensus_steps, self.params['epsilon_bar'])
            sum_norm_sq = self._run_consensus(norm_sq_vals, G_k, num_consensus_steps, self.params['epsilon_bar'])

            # Calculate y0[k] and ||b2[k]|| --- CHECKED
            y0_k = dict()
            norm_b2_k = dict()
            for i in self.nodes.keys():
                y0_k[i] = sum_rayleigh_num[i] / sum_rayleigh_den[i] if sum_rayleigh_den[i] > 1e-9 else 0
                norm_b2_k[i] = np.sqrt(sum_norm_sq[i] * self.N)
                print(f"y0 = {y0_k[i]:.4f}, ||b2]|| = {norm_b2_k[i]:.4f}")

            # --- Steps 4, 5, 6: Nodes update their estimates locally --- CHECKED
            for i, node in self.nodes.items():
                node.update_eigenvalue_estimate(y0_k[i], alpha_k)
                node.update_eigenvector_estimate(norm_b2_k[i])
                print(f"Node {i}: y = {node.y:.4f}, z = {node.z:.4f}, x_i = {node.x_i:.4f}")
            print()

            # Store the estimate from one node (they should all be the same)
            history.append([self.nodes[i].z for i in range(self.N)])

        return history


def load_graph(index, graph_data):
    """Load the next graph from the graph_positions.json file."""

    current_data = graph_data[index]
    node_positions = current_data['positions']
    communication_radius = current_data.get('communication_radius', 0.2)

    # Create graph based on positions and communication radius
    G = nx.Graph()

    # Add all nodes
    for node_id in node_positions.keys():
        G.add_node(int(node_id))

    # Add edges between nodes within communication radius and publish
    node_ids = list(node_positions.keys())
    for i, node_id1 in enumerate(node_ids):
        for node_id2 in node_ids[i+1:]:
            pos1 = node_positions[node_id1]
            pos2 = node_positions[node_id2]
            distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

            if distance <= communication_radius:
                G.add_edge(int(node_id1), int(node_id2))

    return G


if __name__ == '__main__':
    # --- Simulation Parameters ---
    PARAMS = {
        'num_nodes': 5,
        'connection_prob': 1.0,     # p_c: probability a link is active at time k
        'num_iterations': 20,
        'num_consensus_steps': 50,  # Fixed number of steps for consensus rounds
        'epsilon_bar': 0.25,        # Constant step-size for b_i vector, eps < 2/lambda_N <= 1/d_max <= 1/(N-1)
        'alpha0': 1.5,              # Diminishing step-size params for y[k] update
        'beta': 0.51,               # 0.5 < beta <= 1
        'epsilon0': 0.4,            # Diminishing step-size params for b_{2,i} vector
        'gamma': 0.51               # 0.5 < gamma <= 1
    }

    # --- Setup ---
    print("Setting up network...")
    config_path = "/root/ros2_ws/src/ros2_dist_gnn/ros2_dist_gnn/config/graph_positions.json"
    with open(config_path, 'r') as f:
        graph_data = json.load(f)

    ideal_graph = load_graph(0, graph_data)  # Load the first graph as the ideal graph

    network = Network(
        num_nodes=PARAMS['num_nodes'],
        ideal_graph=ideal_graph,
        connection_prob=PARAMS['connection_prob'],
        params=PARAMS
    )

    # --- Run Simulation ---
    print("Running distributed estimation...")
    estimation_history = network.run_simulation(
        num_iterations=PARAMS['num_iterations'],
        num_consensus_steps=PARAMS['num_consensus_steps']
    )
    print("Simulation finished.")

    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    estimation_history = np.array(estimation_history)
    ax.plot(estimation_history, label='Distributed Estimate of $\lambda_2(\overline{L})$', linewidth=2)
    ax.axhline(
        y=network.theoretical_lambda2,
        color='r',
        linestyle='--',
        label='Theoretical $\lambda_2(\overline{L})$'
    )

    ax.set_xlabel('Iteration Index (k)', fontsize=14)
    ax.set_ylabel('Algebraic Connectivity Estimate', fontsize=14)
    ax.set_title('Distributed Estimation of Algebraic Connectivity over a Random Graph', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_xlim(0, PARAMS['num_iterations'])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.show()