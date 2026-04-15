from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    """Utility function which loads a single config from a specified location

    Returns:
        Dict: [Dictionary which holds the config populated from yaml file]
    """
    try:
        if str(path).startswith("s3://"):
            from skrec.util.s3 import get_s3_stream

            config_object = get_s3_stream(path)

            return yaml.safe_load(config_object)
        else:
            with open(path, "r") as fp:
                return yaml.safe_load(fp)
    except FileNotFoundError:
        raise
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config at {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unable to load config at {path}: {e}") from e
