#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfiguracja logowania dla frameworka AI Friendliness Evaluator
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logger(name: str = "ai_friendliness", log_level: Optional[str] = None) -> logging.Logger:
    """Konfiguruje i zwraca logger z odpowiednimi ustawieniami.

    Args:
        name: Nazwa loggera
        log_level: Poziom logowania (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                   Jeśli nie podano, używa wartości z konfiguracji lub domyślnego INFO

    Returns:
        Skonfigurowany obiekt Logger
    """
    # Aby uniknąć cyklicznego importu, importujemy tutaj
    from ..config import get_config

    # Pobieramy konfigurację
    config = get_config()

    # Ustalamy poziom logowania
    if log_level is None:
        log_level = config.get("log_level", "INFO")

    # Konwertujemy string na stałą logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Tworzymy logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Unikamy dodawania wielu handlerów
    if not logger.handlers:
        # Formatowanie logów
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Handler dla stdout
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler dla pliku
        log_dir = Path(config.get("storage_dir", "./data")) / "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
