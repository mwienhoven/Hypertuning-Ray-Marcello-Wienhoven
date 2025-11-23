import tomllib


def load_config(path: str = "./config.toml") -> dict:
    """Load the configuration file config.toml

    Args:
        path (str): Path to the configuration file config.toml. Defaults to "./config.toml".

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)
