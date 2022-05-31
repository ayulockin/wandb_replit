import torch
import torch.nn.functional as F


class SimpleMLPModel(torch.nn.Module):
    """
    Simple one hidden layer based MLP model
    for binary classification.

    Args:
        hidden_size (int): Number of neurons in the hidden layer.
    """
    def __init__(self,
                 hidden_size=10,
                 dropout_rate=0.2):
        super(SimpleMLPModel, self).__init__()
        self.fc1 = torch.nn.Linear(2, hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        dropout = self.dropout(relu)
        output = self.fc2(dropout)
        output = self.sigmoid(output)
        return output


class MLPModel(torch.nn.Module):
    """
    A two hidden layer MLP model for binary classification.

    Args:
        first_hidden_size (int): Number of neurons in the first
            hidden layer.
        second_hidden_size (int): Number of neurons in the second
            hidden layer.
        dropout_rate (float): The amount of dropout to be applied.
            The dropouts are after every hidden layer.
    """
    def __init__(self,
                 first_hidden_size=64,
                 second_hidden_size=32,
                 dropout_rate=0.25):
        super(MLPModel, self).__init__()
        self.input_layer = torch.nn.Linear(2, first_hidden_size)
        self.hidden_layer = torch.nn.Linear(first_hidden_size,
                                            second_hidden_size)
        self.output_layer = torch.nn.Linear(second_hidden_size, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output_layer(x))
        return x