from dotenv import dotenv_values
import wandb
import logging

from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import DictAccessor

def wandb_setup(model_config: dict):
    config = dotenv_values("../.env")
    logging.info(f"Config receieved:\n{config}")
    if 'WANDB_KEY' in list(config.keys()):
        wandb.login(key=config['WANDB_KEY'])
        wandb.init(
            project="DDI-KT-2024",
            config=model_config
        )
        config = wandb.config
    else:
        logging.warning("No key found. Wandb won't record the training process.")
        config = DictAccessor(model_config)
    return config