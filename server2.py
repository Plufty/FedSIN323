from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import os

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    for num_examples, m in metrics:
        print(f"Examples: {num_examples}, Accuracy: {m['accuracy']:.2%}")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Define custom callback to save weights at the end of each round
class SaveWeightsCallback(fl.server.strategy.StrategyCallback):
    def on_end_round(self, strategy: "fl.server.strategy.Strategy") -> None:
        # Save weights at the end of each round
        for i, client_manager in enumerate(strategy.client_managers):
            client_manager.get_weights().save(f"client_{i}_weights.h5")

# Define strategy with the custom callback
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    on_end_rounds=[SaveWeightsCallback()],
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
