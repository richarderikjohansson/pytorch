from torch import nn
import torch


class LinearRegressionModel(nn.Module):
    """
    Linear regression model for Chapter 1
    """

    def __init__(self):
        """Init constructor"""

        # call __init__ from nn.Module
        super().__init__()

        # set random seed
        torch.random.manual_seed(42)

        # initialize parameters with random values and require gradient
        self.bias = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float))
        self.weight = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward model

        Method that computes runs data through
        the neural net

        Args:
            x: Input data

        Returns:
            preds: prediction of the neural net
        """

        # computation of the neural net
        preds = self.weight * x + self.bias
        return preds


class MoonsModel_relu(nn.Module):
    """
    Classifier model with ReLU activation
    """

    def __init__(self):
        """Initialize layers and activation function"""
        super().__init__()

        # layers 1,2,3 and non linear activation function (ReLU)
        self.input = nn.Linear(in_features=2, out_features=10)
        self.hidden1 = nn.Linear(in_features=10, out_features=10)
        self.output = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """Forward pass

        Neural network with activation between layers

        Args:
            x: Input
        """
        # neural network
        forward_pass = self.output(self.relu(self.hidden1(self.relu(self.input(x)))))
        return forward_pass
