from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import matplotlib.pyplot as plt
import numpy as np

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Função para plotar e salvar gráfico
def plot_and_save_metrics(metrics_list, output_path):
    # Extrair as métricas específicas que você deseja plotar (por exemplo, acurácia)
    accuracies = [m["accuracy"] for m in metrics_list]

    # Criar um gráfico de linha simples
    plt.plot(np.arange(1, len(accuracies) + 1), accuracies, label="Acurácia")
    plt.xlabel("Rodada de Treinamento")
    plt.ylabel("Acurácia Ponderada")
    plt.title("Acompanhamento de Acurácia ao Longo do Tempo")

    # Adicionar legendas, grade, etc. conforme necessário

    # Salvar o gráfico como um arquivo PDF
    plt.savefig(output_path, format="pdf")

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Inicializar lista para armazenar métricas
all_metrics = []

# Iniciar servidor Flower com método fit
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

# Realizar rodadas de treinamento
for round_num in range(1, 4):  # Número de rodadas (ajuste conforme necessário)
    metrics = fl.server.server.fit(round_num=round_num)
    all_metrics.extend(metrics)

# Após o término do treinamento, chamar a função para plotar e salvar métricas
output_path = "./grafico.pdf"
plot_and_save_metrics(all_metrics, output_path)

