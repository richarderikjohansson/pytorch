import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch.utils import find_assets, get_logger, save_model, load_model, set_cmap, RdYlBu
from pytorch.models import LinearRegressionModel


class Chapter1:
    """
    Exercise class for Chapter 1
    """

    def __init__(self):

        # use CUDA cores if available, else CPU cores
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # initiate figure directory and logger object
        self.assetsdir = find_assets()
        self.logger = get_logger()
        colors = RdYlBu()

        self.red = colors[0]
        self.blue = colors[-1]
        self.orange = colors[3]
        self.lightblue = colors[-3]

    def e1(self):
        """First exercise of Chapter 1

        In this exercise will data be created using the linear
        regression formula. The data will be split in training and
        testing data (80/20 split) and plot the training and testing
        data for visualization
        """
        self.logger.info("--- Running Exercise 1 ---")

        # ground truths (bias and weight would not be known for a real problem)
        bias = 0.9
        weight = 0.3
        start = 0
        end = 1
        step = 0.005

        # create data set
        X = torch.arange(start, end, step).unsqueeze(dim=1)
        y = weight * X + bias

        # get the split index
        split = int(0.8 * len(X))

        # split data into training and testing sets
        self.X_train = X[:split]
        self.y_train = y[:split]
        self.X_test = X[split:]
        self.y_test = y[split:]

        self.logger.info("Plotting training and testing data\n")

        # plot training and testing data
        plot_predictions(obj=self, figname=self.assetsdir / "ch1e1.png")

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

        # initiating model and printing current parameters
        self.model = LinearRegressionModel()
        params = self.model.state_dict()
        self.logger.info(f"Current parameters:\n {params}\n")

    def e3(self, epochs: int):
        """Third exercise of Chapter 1

        In this exercise we are supposed to develop the training
        loop and also print out how the loss progresses

        Args:
            epochs: Number of times the model should go through the
                training loop
        """
        self.logger.info("--- Running Exercise 3 ---")
        self.logger.info("Creating loss and optimizer functions")

        # set up loss and optimizer functions
        self.loss = nn.L1Loss()
        self.optimizer = torch.optim.SGD(
            params=self.model.parameters(),
            lr=0.01
        )

        self.epochs = epochs
        self.logger.info("Entering training loop")

        training_loss = []
        testing_loss = []

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
                training_loss.append(loss.numpy())
                testing_loss.append(test_loss.numpy())

                if epoch % 20 == 0:
                    print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

        self.logger.info("Training done")
        self.logger.info("Plotting loss\n")
        fig = plt.figure(figsize=(10, 7))

        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(range(epochs), training_loss, color=self.red, label="Training loss")
        ax.plot(range(epochs), testing_loss, color=self.blue, label="Testing loss")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend()
        fig.savefig(self.assetsdir / "ch1e3")

    def e4(self):
        """Forth exercise in Chapter 1

        In this exercise we should evaluate our model by 
        plotting the trained models predictions
        """
        self.logger.info("--- Running Exercise 4 ---")

        # put model into evaluation mode
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(self.X_test)

        self.logger.info("Plotting the predictions\n")

        # plot predictions
        plot_predictions(obj=self, figname=self.assetsdir / "ch1e4.png", predictions=predictions)

    def e5(self, name: str):
        """Fifth exercise in Chapter 1

        In this exercise we are saving and loading 
        our trained model. We are also supposed to ensure
        that our loaded model is giving the same predictions
        as the model we saved

        Args:
            name: name of the model to be saved and loaded. 
                Should end with .pth
        """
        self.logger.info("--- Running Exercise 5 ---")

        # save model
        save_model(model=self.model, name=name)
        self.logger.info(f"Saved {self.model} as {name}")

        # initiate new model
        model = LinearRegressionModel()
        self.logger.info(f"Current parameters of newly loaded model:\n{model.state_dict()}")
        self.logger.info(f"Loading model {name}")

        # load model
        model_loaded = load_model(model, name)
        self.logger.info(f"Current parameters of loaded model:\n{model_loaded.state_dict()}")

        # initiate figure object
        fig = plt.figure(figsize=(10, 7))
        gs = GridSpec(1, 1)

        # initiate matplotlib axis and set title
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("Training, testing data with predictions from trained and loaded model")

        # plot training and testing data
        ax.scatter(self.X_train, self.y_train, c=self.red, s=4, label="Training Data")
        ax.scatter(self.X_test, self.y_test, s=4, c=self.blue, label="Testing Data")

        # put models in evaluation mode
        self.model.eval()
        model_loaded.eval()

        # get predictions
        with torch.inference_mode():
            preds = self.model(self.X_test)
            preds_loaded = model_loaded(self.X_test)

        # plot predictions
        self.logger.info("Plotting")
        ax.scatter(self.X_test, preds, c=self.orange, s=4, label="Predictions")
        ax.plot(self.X_test, preds_loaded, c=self.lightblue, label="Predictions from loaded model", linestyle="-.")
        ax.grid(linestyle="dashed", alpha=0.3)
        ax.legend()

        # save figure and close figure
        fig.savefig(self.assetsdir / "ch1e5.png")
        plt.close()


def plot_predictions(obj,
                     figname,
                     predictions=None):

    train_data = obj.X_train
    train_labels = obj.y_train
    test_data = obj.X_test
    test_labels = obj.y_test

    gs = GridSpec(1, 1)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Traning and testing data")
    ax.scatter(train_data, train_labels, c=obj.red, s=4, label="Training Data")
    ax.scatter(test_data, test_labels, c=obj.blue,  s=4, label="Test Data")

    if predictions is not None:
        ax.scatter(test_data,
                   predictions,
                   s=4,
                   c=obj.lightblue,
                   label="Predictions")

    ax.grid(linestyle="dashed", alpha=0.2)
    ax.legend()
    fig.savefig(figname, transparent=False)
    plt.close()


obj = Chapter1()
obj.e1()
obj.e2()
obj.e3(epochs=300)
obj.e4()
obj.e5(name="ch1_linred.pth")
