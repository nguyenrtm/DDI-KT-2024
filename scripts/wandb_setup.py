from dotenv import dotenv_values
import wandb
import logging

from ddi_kt_2024 import logging_config
from ddi_kt_2024.utils import DictAccessor

def wandb_setup(model_config: dict):
    config = dotenv_values("../.env")
    logging.info(f"Config receieved:\n{config}")
    wandb_available = False
    if 'WANDB_KEY' in list(config.keys()):
        wandb.login(key=config['WANDB_KEY'])
        wandb.init(
            project="DDI-KT-2024",
            config=model_config
        )
        config = wandb.config
        wandb_available = True

        # Update eval values
        wandb.config.update({
            "w_false": eval(config.w_false), 
            "w_advice": eval(config.w_advice),
            "w_effect": eval(config.w_effect),
            "w_mechanism": eval(config.w_mechanism),
            "w_int": eval(config.w_int)
            }, allow_val_change=True)
    else:
        logging.warning("No key found. Wandb won't record the training process.")
        config = DictAccessor(model_config)
    return config, wandb_available