import json


def save_json(obj, path):
    """
    Saves a dict of metrics or training history as JSON.
    """
    # Numpy numbers are converted to plain floats.
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def load_json(path):
    """
    Loads a dict saved by `save_json`.
    """
    with open(path) as f:
        return json.load(f)
