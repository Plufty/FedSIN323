import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from torchvision.datasets import ImageFolder
import torchvision.models as models
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import random


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import numpy as np
import datetime



warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
START = time.time()
MODEL = "alexnet"
DATE_NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, loss, elapsed_time, true_labels, predicted_labels, model_name=MODEL):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}, Loss: {loss:.4f}, Time: {elapsed_time:.2f} seconds')

    # Save the plot as a PDF
    output_dir = output_dir + r'/outputs/' + model_name
    os.makedirs(output_dir, exist_ok=True)
    result_dir = output_dir + '/' + model_name +'_' + DATE_NOW
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f'{model_name}_confusion_matrix.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')

    # Calculate precision, recall, and F1-score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds")



class Net(nn.Module):
    def __init__(self, model_name=MODEL) -> None:
        super(Net, self).__init__()

        if model_name == "alexnet":
            self.model = models.alexnet(weights='DEFAULT')
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, 2)
        elif model_name == "resnet":
            self.model = models.resnet50(weights='DEFAULT')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2) 
        elif model_name == "efficientnet":
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
            num_features = self.model._fc.in_features
            self.model._fc = nn.Linear(num_features, 2)
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train(net, trainloader, epochs, output_dir, model_name=MODEL):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # Lists to store loss and accuracy per epoch
    train_loss = []
    train_accuracy = []
    preds = 0

    for epoch in range(epochs):
        correct, total, total_loss = 0, 0, 0.0
        for inputs, labels in tqdm(trainloader):            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            net.to(DEVICE)

            outputs = net(inputs)

            optimizer.zero_grad()
            
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)
            #correct += (torch.max(outputs.data, 1)[1] == labels.to(DEVICE)).sum().item()
        
        # Calculate accuracy and save loss and accuracy
        accuracy = correct.double() / len(trainloader.dataset)
        epoch_loss = total_loss / len(trainloader.dataset)
        #accuracy = correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2%}")

    # Create and save loss and accuracy plots
    epochs_range = np.arange(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    #plt.plot(epochs_range, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs_range, [acc.cpu().numpy() for acc in train_accuracy], 'r', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plot as a PDF
    output_dir = output_dir + r'/outputs/'  + model_name
    os.makedirs(output_dir, exist_ok=True)
    result_dir = output_dir + '/' + model_name +'_' + DATE_NOW
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f'{model_name}_loss_accuracy_plots.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')



def test(net, testloader, output_dir):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    true_labels = []
    predicted_labels =  []

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):            
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            
            loss += criterion(outputs, labels).item() * inputs.size(0)

            correct += (predicted == labels).sum().item()
            
            #correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            # Collect true and predicted labels for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    end_time = time.time()
    elapsed_time = end_time - START
    accuracy = correct / len(testloader.dataset)
    real_loss = loss / len(testloader.dataset)

    print(f"Test Loss: {real_loss:.4f}, Test Accuracy: {accuracy:.2%}")

    # Save the confusion matrix and accuracy
    class_names = ["benign", "malignant"]
    save_confusion_matrix(true_labels, predicted_labels, class_names, output_dir, accuracy, loss, elapsed_time, true_labels, predicted_labels)


    return real_loss, accuracy



def load_data():
    # Load the breast cancer dataset (modify the paths accordingly)
    input_size = 224
    data_transforms = {
        'transform': transforms.Compose([
		transforms.Resize([input_size, input_size], antialias=True),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	    ])
    }               
                
    trainset = ImageFolder("./data/train", transform=data_transforms['transform'])
    testset = ImageFolder("./data/test", transform=data_transforms['transform'])
    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


net = Net().to(DEVICE)
trainloader, testloader = load_data()

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        parameters = []
        for _, val in net.state_dict().items():
            parameters.append(val.cpu().numpy())
        return parameters

    def set_parameters(self, parameters):
        state_dict = net.state_dict()
        for key, param in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(param)
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1, output_dir="./")
        
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader, "./") 
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="192.168.217.129:8080",
    client=FlowerClient(),
)
