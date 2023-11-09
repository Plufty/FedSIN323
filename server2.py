from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import matplotlib.pyplot as plt
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

def plot_and_save_metrics(metrics_list, output_path):
    accuracies = [m["accuracy"] for m in metrics_list]

    plt.plot(np.arange(1, len(accuracies) + 1), accuracies, label="Acurácia")
    plt.xlabel("Rodada de Treinamento")
    plt.ylabel("Acurácia Ponderada")
    plt.title("Acompanhamento de Acurácia ao Longo do Tempo")


    plt.savefig(output_path, format="pdf")

strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

all_metrics = []


server = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

for round_num in range(1, 4): 
    metrics = server.fit(round_num=round_num)
    all_metrics.extend(metrics)


output_path = "./grafico.pdf"
plot_and_save_metrics(all_metrics, output_path)

