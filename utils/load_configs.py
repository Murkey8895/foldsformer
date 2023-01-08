import yaml


def get_configs(filepath):
    with open(filepath, "rb") as f:
        configs = yaml.safe_load(f)
    return configs
