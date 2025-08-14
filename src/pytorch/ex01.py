import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch.utils import find_assets, get_logger, save_model


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.random.manual_seed(42)
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.weight = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


class Chapter2:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assetsdir = find_assets()
        self.logger = get_logger()

    def e1(self):
        """First exercise of Chapter 1

        In this exercise will data be created using the linear
        regression formula. The data will be split in training and
        testing data (80/20 split) and plot the training and testing
        data for visualization
        """
        self.logger.info("--- Running Exercise 1 ---")
        bias = 0.9
        weight = 0.3
        start = 0
        end = 1
        step = 0.005

        X = torch.arange(start, end, step).unsqueeze(dim=1)
        y = weight * X + bias

        split = int(0.8 * len(X))

        self.X_train = X[:split]
        self.y_train = y[:split]
        self.X_test = X[split:]
        self.y_test = y[split:]

        self.logger.info("Plotting training and testing data\n")
        plot_predictions(self.X_train,
                         self.y_train,
                         self.X_test,
                         self.y_test,
                         figname=self.assetsdir / "ch1e1.png")

    def e2(self):
        """Second exercise of Chapter 1

        In this we shall construct a linear regression model with
        randomized parameters. We should also construct a forward
        method that gives the prediction from the input and
        parameters. We should the initiate this model and print out
        the current parameters.
        """
        self.logger.info("--- Running Exercise 2 ---")
        self.logger.info("Initiating the model")
        self.model = LinearRegressionModel()
        params = self.model.state_dict()
        self.logger.info(f"Current parameters:\n {params}\n")

    def e3(self, epochs):
        """Third exercise of Chapter 1

        In this exercise we are supposed to develop the training
        loop and also print out how the loss progresses

        Args:
            epochs ([TODO:type]): [TODO:description]
        """
        self.logger.info("--- Running Exercise 3 ---")
        self.logger.info("Creating loss and optimizer functions")
        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.01
        )

        self.epochs = epochs
        self.logger.info("Entering training loop")

        # training loop
        for epoch in range(self.epochs):

            # set model into training mode
            self.model.train()

            # forward pass through network
            y_pred = self.model(self.X_train)

            # calculate loss
            loss = self.loss(y_pred, self.y_train)

            # set optimizer to zero gradient
            self.optimizer.zero_grad()

            # back propagate
            loss.backward()

            # step the optimizer
            self.optimizer.step()

            # enter testing mode
            self.model.eval()
            with torch.inference_mode():
                test_pred = self.model(self.X_test)
                test_loss = self.loss(test_pred, self.y_test)

                if epoch % 20 == 0:
                    print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

        self.logger.info("Training done\n")

    def e4(self):
        self.logger.info("--- Running Exercise 4 ---")
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(self.X_test)

        self.logger.info("Plotting the predictions\n")
        plot_predictions(
            train_data=self.X_train,
            train_labels=self.y_train,
            test_data=self.X_test,
            test_labels=self.y_test,
            figname=self.assetsdir / "ch1e4.png",
            predictions=predictions
        )

    def e5(self, name):
        self.logger.info("--- Running Exercise 5 ---")
        save_model(model=self.model, name=name)
        self.logger.info(f"Saved {self.model} as {name}\n")



def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     figname,
                     predictions=None):

    gs = GridSpec(1, 1)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Traning and testing data")
    ax.scatter(train_data, train_labels, c="dimgray", s=4, label="Training Data")
    ax.scatter(test_data, test_labels, c="orange",  s=4, label="Test Data")

    if predictions is not None:
        ax.scatter(test_data,
                   predictions,
                   c="tomato",
                   s=4,
                   label="Predictions")

    ax.legend()
    fig.savefig(figname, transparent=False)
    plt.close()


obj = Chapter2()
obj.e1()
obj.e2()
obj.e3(epochs=300)
obj.e4()
obj.e5(name="ch1_linred.pth")
