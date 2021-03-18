import yaml
import copy
from pathlib import Path
import json


def load_config(config_path):
    if isinstance(config_path, Path):
        config_path = str(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.endswith('yml') or config_path.endswith('yaml'):
            config = yaml.load(f, Loader=yaml.SafeLoader)
        elif config_path.endswith('json'):
            config = json.load(f)
    
    return config

def merge_subcfg(cfg, default_args=None):
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')

    args = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return args