"""
Read yaml file
"""
import logging
import yaml

from ddi_kt_2024 import logging_config

def get_yaml_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f.read())

    return dict(config)
