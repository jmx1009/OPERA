"""Configuration shared by the modeling and library-learning stages."""
import os
from pathlib import Path

from omegaconf import OmegaConf


MODEL = "gpt-4o"
SERVICE = "openai"


def validate_model(model, service=None):
    """Check the manuscript model before opening a client or starting a run."""
    if model != MODEL or service not in (None, SERVICE):
        raise ValueError("OPERA uses GPT-4o through the OpenAI service.")


def training_config():
    return OmegaConf.load(os.environ.get("OPERA_TRAIN_CONFIG", Path(__file__).resolve().parents[1] / "train_config.yaml"))
