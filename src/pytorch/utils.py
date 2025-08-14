from pathlib import Path
import logging
import torch


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


def save_model(model, name):
    modeldir = Path("models")
    basedir = Path(__file__)
    savedir = basedir.parent / modeldir

    if not savedir.exists():
        savedir.mkdir()

    torch.save(obj=model.state_dict(),
               f=savedir / name)


def load_model(name):
    pass
