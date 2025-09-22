#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import re
from collections import defaultdict

import networkx as nx
import numpy as np
import rclpy
import rclpy.qos
import torch
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String, Float32

from ros2_dist_gnn_msgs.msg import Discovery
from ros2_dist_gnn.utils.led_matrix import LEDMatrix
from stochastic_pi_msgs.msg import ConsensusMessage

comm_qos = rclpy.qos.QoSProfile(
    history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=rclpy.qos.QoSReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

fresh_qos = rclpy.qos.QoSProfile(
    history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
    durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class MyNode(Node):

    def __init__(self):
        super().__init__('gnn_node')

        # Unique identifier for the node. # DOC: We assume all nodes have the same format of the name.
        self.node_name = self.get_namespace().strip("/")
        if self.node_name == '/':
            self.node_name = 'node_0'

        if not re.match(r'^[a-zA-Z_]+\d+$', self.node_name):
            self.get_logger().error(f"Node name should be in the format <string>_<integer. Got: {self.node_name}")
            raise ValueError(f"Invalid node name format: {self.node_name}")

        self.node_prefix = self.node_name.split("_")[0] + "_"
        self.node_id = int(self.node_name.split("_")[-1])
        self.get_logger().info(f'GNN Node {self.node_name} has been started.')

        self.value = torch.Tensor()  # The current representation of the node.
        self.layer = 0  # The current layer of the GNN being processed.
        self.received_msg = defaultdict(lambda: defaultdict(dict))  # Consensus values received from neighbors, indexed by iteration number.
                                               # Also used for synchronization.

        self.round_counter = 0
        self.local_subgraph = nx.Graph()
        self.latest_discovery_msg = Discovery()
        self.active_neighbors = []

        # Load parameters
        self.declare_parameter("num_nodes", 1)
        self.declare_parameter("comm_radius", 2.0)
        self.declare_parameter('epsilon_bar', 0.8)  # Consensus step size for the average graph.
        self.declare_parameter('epsilon0', 0.4)     # Initial consensus step size for B2. (7, 20)
        self.declare_parameter('alpha0', 1.5)       # Initial step size for y. (11)
        self.declare_parameter('gamma', 0.5)       # Decay rate for epsilon. (16)
        self.declare_parameter('beta', 0.5)        # Decay rate for alpha. (16)
        self.declare_parameter('num_consensus_steps', 25)
        self.declare_parameter('num_pi_steps', 10)

        self.num_nodes = self.get_parameter("num_nodes").get_parameter_value().integer_value
        self.communication_radius = self.get_parameter("comm_radius").get_parameter_value().double_value
        self.epsilon_bar = self.get_parameter('epsilon_bar').get_parameter_value().double_value
        self.epsilon0 = self.get_parameter('epsilon0').get_parameter_value().double_value
        self.alpha0 = self.get_parameter('alpha0').get_parameter_value().double_value
        self.gamma = self.get_parameter('gamma').get_parameter_value().double_value
        self.beta = self.get_parameter('beta').get_parameter_value().double_value
        self.num_consensus_steps = self.get_parameter('num_consensus_steps').get_parameter_value().integer_value
        self.num_pi_steps = self.get_parameter('num_pi_steps').get_parameter_value().integer_value

        # Define callback groups. Using callback groups ensures the proper use of resources thanks to multi-threading.
        # All callbacks assigned to a MutuallyExclusiveCallbackGroup will be executed in the same thread, but in
        # parallel to callbacks in other callback groups.
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()  # Main timer loop callback group.
        self.consensus_cb_group = MutuallyExclusiveCallbackGroup()  # Exclusively for message passing.
        self.pooling_cb_group = MutuallyExclusiveCallbackGroup()  # Exclusively for pooling.
        self.sub_cb_group = MutuallyExclusiveCallbackGroup()  # Exclusively for subscribing to other topics.

        # Create publishers for sending values to neighbors.
        self.consensus_pubs = {f"{self.node_prefix}{i}": self.create_publisher(ConsensusMessage, f"/{self.node_prefix}{i}/consensus", comm_qos) for i in range(self.num_nodes) if i != self.node_id}
        self.lambda_pub = self.create_publisher(Float32, "lambda2", fresh_qos)

        # Create subscribers for receiving values from neighbors and neighbor positions.
        self.consensus_sub = self.create_subscription(ConsensusMessage, "consensus", self.receive_consensus, comm_qos, callback_group=self.consensus_cb_group)

        # Subscriber for receiving the whole graph.
        #   This is used only in development to test the functionality using the
        #   predefined full graph generated by graph_generator.py without robot
        #   position data.
        self.declare_parameter("dev_mode", False)
        self.dev_mode = self.get_parameter("dev_mode").get_parameter_value().bool_value
        if self.dev_mode:
            self.graph_sub = self.create_subscription(String, '/graph_topic', self.graph_cb, fresh_qos, callback_group=self.sub_cb_group)
        else:
            self.discovery_sub = self.create_subscription(Discovery, 'discovery', self.discovery_cb, fresh_qos, callback_group=self.sub_cb_group)

        self.main_loop = self.create_timer(0.1, self.compute_pi, callback_group=self.timer_cb_group)

        # Initialize the LED matrix if available.
        self.led = LEDMatrix()

        self.get_logger().debug("Node initialized.")

    def graph_cb(self, msg):
        G = nx.from_graph6_bytes(bytes(msg.data.strip(), "ascii"))
        lambda2 = nx.laplacian_spectrum(G)[1]
        self.get_logger().debug(f"Received graph {msg.data} with algebraic connectivity λ₂: {lambda2:.4f}")

        self.local_subgraph: nx.Graph = G.subgraph([self.node_id] + list(G.neighbors(self.node_id)))

    def discovery_cb(self, msg: Discovery):
        # DOC: Change how discovery messages are processed to build the local subgraph.
        self.latest_discovery_msg = msg

    def get_neighbors(self):
        self.active_neighbors = []

        if self.dev_mode:
            # We know the whole graph in development mode.
            if self.local_subgraph.number_of_nodes() > 0:
                self.active_neighbors = [f"{self.node_prefix}{i}" for i in self.local_subgraph.neighbors(self.node_id)]
        else:
            node_ids = [self.node_id] + list(self.latest_discovery_msg.neighbor_ids)
            node_pos = [self.latest_discovery_msg.own_position] + list(self.latest_discovery_msg.neighbor_positions)

            self.local_subgraph = nx.Graph()
            self.local_subgraph.add_nodes_from(node_ids)

            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    distance = ((node_pos[i].x - node_pos[j].x)**2 + (node_pos[i].y - node_pos[j].y)**2)**0.5
                    if distance <= self.communication_radius:
                        self.local_subgraph.add_edge(node_ids[i], node_ids[j])

            self.active_neighbors = [f"{self.node_prefix}{i}" for i in self.local_subgraph.neighbors(self.node_id)]

        assert nx.is_connected(self.local_subgraph), "Local subgraph is not connected! This should never happen."
        ready = len(self.active_neighbors) > 0

        if ready:
            self.get_logger().info(f"Active neighbors: {self.active_neighbors}")
        return ready

    def get_initial_features(self):
        rng = np.random.default_rng()
        self.x_i = rng.standard_normal((1,))
        self.y = rng.random((1,))
        self.z = rng.random((1,))

    def run_consensus(self, initial_value, consensus_id):
        local_value = initial_value

        for step in range(self.num_consensus_steps):
            # Send the current node representation to neighbors.
            self.send_consensus(step, local_value, consensus_id)

            # Wait until values are received from all neighbors.
            wait_time_start = time.time()
            while len(self.received_msg[consensus_id][step]) < len(self.active_neighbors):
                if time.time() - wait_time_start > 2.0:  # 2 seconds timeout
                    self.get_logger().warn(f"Timeout waiting for messages at step {step}. Proceeding with available messages.")
                    break
                self.get_logger().debug(f"Waiting for messages at step {step}...", throttle_duration_sec=1.0)
                time.sleep(0.01)

            # Update the node's representation using the GNN layer.
            neighbor_values = list(self.received_msg[consensus_id][step].values())
            local_value = (local_value + sum(neighbor_values)) / (1 + len(neighbor_values))

            self.get_logger().debug(f"Updated step {step}.")

        if "ray" in consensus_id:
            local_value *= self.num_nodes  # Scale up the aggregated value for Rayleigh ratio.

        return local_value

    def send_consensus(self, step: int, value: np.ndarray, id: str):
        msg = ConsensusMessage()
        msg.sender = self.node_name
        msg.step = step
        msg.id = id
        msg.data = value.flatten().tolist()  # Flatten tensor to 1D list
        msg.shape = list(value.shape)  # Store original shape

        for neighbor in self.active_neighbors:
            self.consensus_pubs[neighbor].publish(msg)
            self.get_logger().debug(f"Sent message to {neighbor} at step {step}")

    def receive_consensus(self, msg: ConsensusMessage):
        self.get_logger().debug(f"Received message from {msg.sender} at step {msg.step}")
        # Reconstruct tensor from flattened data and shape
        data = np.array(msg.data).reshape(tuple(msg.shape))
        self.received_msg[msg.id][msg.step][msg.sender] = data

    def neighbor_position_cb(self, msg):
        pass

    def get_step_sizes(self, k):
        alpha_k = self.alpha0 / ((k + 1) ** self.beta)
        epsilon_k = self.epsilon0 / ((k + 1) ** self.gamma)
        return alpha_k, epsilon_k

    def compute_pi(self):
        ready = self.get_neighbors()
        if not ready:
            self.get_logger().info("Waiting for neighbors...", throttle_duration_sec=1.0)
            return

        self.round_counter += 1
        self.get_logger().info(f"Round {self.round_counter} started.")

        self.get_initial_features()

        # Main stochastic PI logic.
        for k in range(self.num_pi_steps):
            alpha_k, epsilon_k = self.get_step_sizes(k)

            # --- Step 1: Consensus for m[k] ---
            m_k = self.run_consensus(self.x_i, f"{k}_m")
            self.get_logger().info(f"Consensus m[{k}] = {m_k[0]:.4f}")

            # --- Step 2: Fetch neighbor x_j and compute b vectors ---
            last_step = max(self.received_msg[f"{k}_m"].keys())
            sum_term = sum(xj - self.x_i for xj in self.received_msg[f"{k}_m"][last_step].values())
            b_i = self.x_i + self.epsilon_bar * sum_term - m_k
            b2_i = self.x_i + epsilon_k * sum_term - m_k
            self.get_logger().info(f"Computed b_i = {b_i[0]:.4f}, b2_i = {b2_i[0]:.4f}")
            del self.received_msg[f"{k}_m"]  # Clear memory

            # --- Step 3: Consensus for Rayleigh Ratio and Norm ---
            initial_value = np.array([self.x_i * b_i, self.x_i**2, b2_i**2])
            new_value = self.run_consensus(initial_value, f"{k}_ray")
            sum_ray_num, sum_ray_den, sum_norm_sq = new_value

            y0_k = sum_ray_num / sum_ray_den if sum_ray_den > 1e-9 else np.array([0])
            norm_b2_k = np.sqrt(sum_norm_sq)
            self.get_logger().info(f"Consensus Rayleigh Ratio = {y0_k[0]:.4f}, Norm = {norm_b2_k[0]:.4f}")
            del self.received_msg[f"{k}_ray"]  # Clear memory

            # --- Steps 4, 5, 6: Local State Updates ---
            self.y = self.y + alpha_k * (y0_k - self.y)
            self.z = (1 - self.y) / self.epsilon_bar
            if norm_b2_k > 1e-9:
                self.x_i = b2_i / norm_b2_k

            lambda2 = self.z[0]
            self.lambda_pub.publish(Float32(data=lambda2))

            # Update LED color based on the estimated value.
            led_color = LEDMatrix.from_colormap(LEDMatrix.interp(lambda2, 0.0, self.num_nodes, 0.0, 1.0), color_space='hsv', cmap_name="jet")
            led_color = (led_color[0], led_color[1], led_color[2] * 0.5)  # Full brightness
            self.led.set_all(led_color)

            # Delete messages from previous iteration to save memory.

        self.get_logger().info(f"Node computed graph value {lambda2:.3f} in round {self.round_counter}.\n")

    def destroy_node(self):
        super().destroy_node()
        self.led.exit()


def main(args=None):
    rclpy.init(args=args)

    gnn_node = MyNode()

    executor = MultiThreadedExecutor()
    executor.add_node(gnn_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        gnn_node.get_logger().info("KeyboardInterrupt received. Shutting down...")
    finally:
        # Destroy the node explicitly
        gnn_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()