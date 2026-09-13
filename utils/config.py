from pathlib import Path
from typing import Any, Dict
import logging

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    config_file = Path(config_path)
    logging.getLogger(__name__).debug("checking config path: %s", config_file.resolve())

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config format in: {config_path}")

    return cfg