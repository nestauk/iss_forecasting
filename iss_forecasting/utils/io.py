"""Loading and saving files utils"""
import pickle
import pathlib


def save_pickle(dir_path: pathlib.PosixPath, filename: str, obj):
    """Save python object as pickle file

    Args:
        dir_path: Path of directory to save pickle file to
        filename: Name to give pickle file
        obj: Python object to save as pickle file
    """
    with open(dir_path / f"{filename}.pickle", "wb") as op:
        pickle.dump(obj, op)


def load_pickle(dir_path: pathlib.PosixPath, filename: str):
    """Load pickle file

    Args:
        dir_path: Path of directory containing pickle file
        filename: Name of pickle file to load
    """
    with open(dir_path / filename, "rb") as ip:
        return pickle.load(ip)
