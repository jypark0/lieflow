import logging
from pathlib import Path
from typing import Callable

import numpy as np
import torch

import wandb


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


class WandBLogger:
    """WandB Logger. Log all episodes results using global_step."""

    def __init__(
        self,
        wandb_config,
        **wandb_kwargs,
    ):
        self.run = wandb.init(config=wandb_config, **wandb_kwargs)

        # Add more keys to wandb.config
        self.run.config.save_path = Path(self.run.dir).resolve()
        self.run.config.id = self.run.id

    def watch_model(self, model):
        self.run.watch(model, log=None, log_graph=True)

    def write(self, result: dict, timestamp: dict = {}) -> None:
        self.run.log({**result, **timestamp})

    def log_all(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        # Log all values in result
        result = {f"{prefix}/{k}": v for k, v in result.items()}

        self.write(result, timestamp)

    def log_mean(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        # Log mean values in result
        result = {f"{prefix}/{k}": np.mean(v) for k, v in result.items()}

        self.write(result, timestamp)

    def log_last(self, prefix: str, result: dict, timestamp: dict = {}) -> None:
        # Log last value only
        result = {f"{prefix}/{k}": v[-1] for k, v in result.items()}

        self.write(result, timestamp)

    def log_image(self, prefix: str, img: np.ndarray, timestamp: dict = {}) -> None:
        self.write({prefix: wandb.Image(img)}, timestamp)

    def save(
        self,
        timestamp: dict,
        checkpoint_fn: Callable[[None], None],
        is_best: bool,
        epoch_key: str = "epoch",
        **kwargs,
    ) -> None:
        if checkpoint_fn:
            checkpoint_fn(save_path=Path(self.run.dir), is_best=is_best, **kwargs)

            # Save timestamp
            timestamp_filename = "timestamp.pt"
            if is_best:
                timestamp_filename = "timestamp-best.pt"
                self.write({"info/saved_best": timestamp[epoch_key]}, timestamp)

            torch.save(
                timestamp,
                Path(self.run.dir) / timestamp_filename,
            )

    def close(self):
        self.run.finish()
