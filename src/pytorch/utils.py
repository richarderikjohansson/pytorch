from pathlib import Path
import logging
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def find_assets():
    """Helper function to find assets directory for docs
    """
    fp = Path(__file__)
    for parent in fp.parents:
        docdir = parent / "docs"
        if docdir.exists():
            assetsdir = docdir / "assets"
            return assetsdir


def get_logger() -> logging.Logger:
    """Initiate logger object

    Returns:
        logger
    """
    logger = logging.getLogger("runtime_logger")
    if not logger.hasHandlers():  # Prevent adding handlers multiple times
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_format)

        logger.addHandler(console_handler)
        logger.propagate = False
    return logger


def save_model(model: torch.nn.Module, name: str):
    """Function to save PyTorch model

    Args:
        model: Model object
        name: Filename that model is saved to
    """
    modeldir = Path("models")
    basedir = Path(__file__)
    savedir = basedir.parent / modeldir

    if not savedir.exists():
        savedir.mkdir()

    torch.save(obj=model.state_dict(),
               f=savedir / name)


def load_model(model: torch.nn.Module, name: str):
    """Function to load model

    Args:
        model: Model object to load parameters to
        name: Filename of the model to be loaded
    """
    modeldir = Path("models")
    basedir = Path(__file__)
    savedir = basedir.parent / modeldir
    modelfile = savedir / name

    if modelfile.exists():
        model.load_state_dict(torch.load(modelfile))

    return model


def set_cmap(fig):
    cmap = "RdYlBu"
    colormap = plt.get_cmap(cmap)
    n = len(fig.axes)
    colors = colormap(np.linspace(0, 1, n))
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    mpl.rcParams["image.cmap"] = cmap


def RdYlBu():
    colors = [
        "#a50026",
        "#d73027",
        "#f46d43",
        "#fdae61",
        "#fee090",
        "#ffffbf",
        "#e0f3f8",
        "#abd9e9",
        "#74add1",
        "#4575b4",
        "#313695"
    ]
    return colors


def plot_decision_boundary(ax, model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    ax.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc
