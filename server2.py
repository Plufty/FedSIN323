from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import os

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by the number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    for num_examples, m in metrics:
        print(f"Examples: {num_examples}, Accuracy: {m['accuracy']:.2%}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Function to save weights
def save_weights(client_manager, round_num):
    for i, client_manager in enumerate(client_manager):
        client_manager.get_weights().save(f"client_{i}_round_{round_num}_weights.h5")

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

# Custom logic for saving weights at the end of each round
round_num = 0
while round_num < strategy.config.num_rounds:
    # Run one round of the strategy
    strategy.next_round()
    
    # Save weights at the end of each round
    save_weights(strategy.client_managers, round_num)
    
    round_num += 1
