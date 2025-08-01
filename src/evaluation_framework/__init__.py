# AI Friendliness Evaluator Framework
# Copyright (c) 2025

__version__ = '0.1.0'

from .config import get_config, set_config_value
from .utils.logging import setup_logger

logger = setup_logger()

def get_version():
    """Zwraca wersję frameworka."""
    return __version__
