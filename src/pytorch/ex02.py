import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from pytorch.utils import get_logger, find_assets, RdYlBu, plot_decision_boundary
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pytorch.models import MoonsModel_relu
import torchmetrics


class Chapter2:
    def __init__(self):
        # device agnostic code
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = get_logger()
        self.assetsdir = find_assets()
        colors = RdYlBu()
        self.red = colors[0]
        self.blue = colors[-1]
        self.orange = colors[3]
        self.cmap = plt.cm.RdYlBu

        torch.manual_seed(42)

    def e1(self, samples, noise):
        """Chapter 2 exercise 1

        - Make a binary classification dataset with sklearns make_moons()
        method:

        * For consistency, the dataset should have 1000 samples and
        a randomstate of 42

        * Turn the data into PyTorch tensors. Split data into 80/20 training
        and testing sets
        """
        self.logger.info("Running exercise 1")
        self.samples = samples

        # load toy dataset
        self.X, self.y = make_moons(
            n_samples=samples, noise=noise, random_state=42)
        # put dataset into pandas dataframe (no need really, only for easier visualization)
        self.df = pd.DataFrame(
            {"X1": self.X[:, 0], "X2": self.X[:, 1], "y": self.y})

        self.logger.info("Plotting loaded dataset")
        plot_make_moons(obj=self)

        # turn input data and label into tensors with type 'float'
        self.logger.info("Turn input and labels into tensors")
        self._X = torch.from_numpy(self.X).type(torch.float)
        self._y = torch.from_numpy(self.y).type(torch.float)

        # set training and testing sets
        self.logger.info(
            "Divide dataset into 80/20 train test split, randomly")
        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(
            self._X, self._y, test_size=0.2, random_state=42
        )
        self.logger.info(f"Length of training set: {len(self.X_train)}")
        self.logger.info(f"Length of test set: {len(self.X_test)}\n")

    def e2(self):
        """Chapter 2 exercise 2

        - Build a model by subclassing nn.Module that incorporates non-linear
        activation functions and is capable of fitting the data from exercise 1.
        """
        self.logger.info("Running exercise 2")
        self.logger.info("Plotting ReLU activation fuction")
        plot_RelU(assetsdir=self.assetsdir)

        # set model
        self.logger.info("Setting model to MoonModel_relu\n")
        self.model_relu = MoonsModel_relu().to(self.device)

    def e3(self):
        """
        Setup a binary classification compatible loss function
        and optimizer to use when training the model.
        """

        self.logger.info("Running exercise 3")

        # setting loss and optimizer functions
        self.logger.info("Defining loss and optimizer functions")
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer_relu = torch.optim.SGD(
            self.model_relu.parameters(), lr=0.1)

    def e4(self, epochs: int):
        """
        Create a training and testing loop to fit the model you created
        in 2 to the data you created in 1.

        Args:
            epochs: Number of epochs that the model will train for
        """
        self.logger.info("Running exercise 4")
        self.epochs = epochs

        # training and accuracy arrays
        training_loss_array = []
        training_acc_array = []

        testing_loss_array = []
        testing_acc_array = []

        eps = []

        # training loop
        for epoch in range(epochs):
            # set model in training mode
            self.model_relu.train()

            # forward pass
            logits = self.model_relu(self.X_train).squeeze()
            y_pred = torch.round(torch.sigmoid(logits))

            # calculate loss and accuracy
            loss = self.loss_fn(logits, self.y_train)

            # zero grad
            self.optimizer_relu.zero_grad()

            # back propagate
            loss.backward()

            # optimizer step
            self.optimizer_relu.step()

            # evaluate
            self.model_relu.eval()
            with torch.inference_mode():
                # get logit
                logit_train = self.model_relu(self.X_train).squeeze()
                logit_test = self.model_relu(self.X_test).squeeze()

                # transform to probability
                y_train = torch.round(torch.sigmoid(logit_train))
                y_test = torch.round(torch.sigmoid(logit_test))

                # get accuracy
                acc_training = torchmetrics.functional.accuracy(
                    preds=y_train,
                    target=self.y_train,
                    task="binary",
                ).numpy()
                acc_test = torchmetrics.functional.accuracy(
                    preds=y_test,
                    target=self.y_test,
                    task="binary",
                ).numpy()

                # get loss
                loss_train = self.loss_fn(logit_train, self.y_train)
                loss_test = self.loss_fn(logit_test, self.y_test)

                # print out every 10 epoch
                if epoch % 10 == 0:
                    print(f"Epoch: {epoch}")
                    stdout = f"""Training loss {loss_train:.3f} | Training accuracy {acc_training:.3f}
Testing loss {loss_test:.3f} | Testing accuracy {acc_test:.3f}\n"""
                    print(stdout)

                # append lists for plotting
                training_loss_array.append(loss_train.numpy())
                testing_loss_array.append(loss_test.numpy())
                training_acc_array.append(acc_training * 100)
                testing_acc_array.append(acc_test * 100)
                eps.append(epoch)

            if acc_training >= 0.99:
                break

        # figure object for training/testing loss and accuracy
        fig = plt.figure()
        gs = GridSpec(nrows=1, ncols=2)

        # axes
        left = fig.add_subplot(gs[0, 0])
        right = fig.add_subplot(gs[0, 1])

        # plot loss
        left.set_title(f"Loss after {eps[-1]} Epochs")
        left.plot(eps, training_loss_array,
                  color=self.red, label="Training loss")
        left.plot(eps, testing_loss_array,
                  color=self.blue, label="Testing loss")
        left.grid(linestyle="dashed", alpha=0.3)
        left.set_xlabel("Epoch")
        left.set_ylabel("Loss")
        left.legend()

        # plot accuracy
        right.set_title(f"Accuracy after {eps[-1]} Epochs")
        right.plot(eps, training_acc_array, color=self.red,
                   label="Training accuracy")
        right.plot(eps, testing_acc_array, color=self.blue,
                   label="Testing accuracy")
        right.grid(linestyle="dashed", alpha=0.3)
        right.set_xlabel("Epoch")
        right.set_ylabel("Accuracy [%]")
        right.legend()

        plt.tight_layout()
        fig.savefig(self.assetsdir / "ch2e4a.png")
        plt.close()

    def e5(self):
        self.logger.info("Running exercise 5")
        self.logger.info("Plotting decision boundary")

        # decision boundary figure
        fig = plt.figure()
        gs = GridSpec(nrows=1, ncols=2)

        left = fig.add_subplot(gs[0, 0])
        right = fig.add_subplot(gs[0, 1])

        left.set_title("Training data")
        plot_decision_boundary(left, self.model_relu,
                               self.X_train, self.y_train)
        right.set_title("Testing data")
        plot_decision_boundary(right, self.model_relu,
                               self.X_test, self.y_test)
        fig.savefig(self.assetsdir / "ch2e5.png")
        plt.close()

    def e6(self):
        self.logger.info("Running exercise 6")
        X = torch.arange(-7, 7, 0.01)
        own_tahn = tanh(X)
        f = nn.Tanh()
        torch_tahn = f(X)

        self.logger.info("Plotting Tahn functions")
        fig = plt.figure()
        gs = GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("Own Tahn function and PyTorch version")
        ax.set_ylim([-4, 4])
        ax.plot(
            X.numpy(),
            (own_tahn.numpy() - torch_tahn.numpy()),
            color="grey",
            alpha=0.6,
            label=r"$\Delta$"
        )
        ax.plot(X.numpy(),
                own_tahn.numpy(),
                color=self.red,
                label="Own Tahn function",
                linestyle="solid",
                linewidth=3)
                
        ax.plot(X.numpy(),
                torch_tahn.numpy(),
                color=self.orange,
                label="PyTorch Tahn function",
                linestyle="dashdot")
        ax.legend()
        ax.grid(linestyle="dashed", alpha=0.3)
        fig.savefig(self.assetsdir / "ch2e6.png")


def plot_make_moons(obj):
    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(1, 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(x=obj.X[:, 0], y=obj.X[:, 1], c=obj.y, cmap=plt.cm.RdYlBu)
    ax.set_title("Moons dataset from sklearn.datasets.make_moons()")
    fig.savefig(obj.assetsdir / "ch2e1.png")


def plot_RelU(assetsdir):
    x = torch.arange(start=-5, end=5, step=0.1)
    rel = nn.ReLU()
    c = RdYlBu()
    y = rel(x)

    plt.set_cmap("RdYlBu")

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, y, color=c[0], label="ReLU activation function")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.grid(linestyle="dashed", alpha=0.3)
    ax.legend()
    fig.savefig(assetsdir / "ch2e2.png")


def tanh(x: torch.Tensor) -> torch.Tensor:
    y = (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    return y


obj = Chapter2()
obj.e1(samples=1000, noise=0.03)
obj.e2()
obj.e3()
obj.e4(epochs=1000)
obj.e5()
obj.e6()
