#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Konfiguracja frameworka AI Friendliness Evaluator

Ten moduł zapewnia scentralizowane zarządzanie konfiguracją dla całego frameworka.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, Optional

# Ładujemy zmienne środowiskowe z pliku .env jeśli istnieje
load_dotenv()

# Domyślna konfiguracja
DEFAULT_CONFIG = {
    # Ogólne ustawienia
    "log_level": "INFO",
    "storage_dir": "./data",

    # Ustawienia API
    "api": {
        "host": "localhost",
        "port": 8000,
        "debug": False,
        "api_key_required": False,
        "allowed_origins": ["*"],
    },

    # Ustawienia oceny
    "evaluation": {
        "friendliness_threshold": 5.0,
        "default_rubrics": [
            "emotion_recognition",
            "empathic_response",
            "consistency",
            "personalization",
            "ethical_alignment"
        ],
        "rubrics_dir": "./rubrics",
        "scenarios_dir": "./scenarios",
    },

    # Ustawienia modelu
    "models": {
        "differential": {
            "use_enhanced_nonlinear": False,
            "use_mcmc": False,
            "prediction_days": 30,
            "optimizer": "BFGS"
        }
    },

    # Integracje
    "integrations": {
        "hume_ai": {
            "enabled": False,
            "api_key": ""
        },
        "deep_eval": {
            "enabled": False,
            "api_key": ""
        }
    }
}

# Globalna konfiguracja
_config = DEFAULT_CONFIG.copy()

def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Łączy dwie struktury konfiguracyjne, nadpisując wartości z override."""
    result = base.copy()

    for key, value in override.items():
        if (
            key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)
        ):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Ładuje konfigurację z pliku JSON i/lub zmiennych środowiskowych."""
    global _config

    # Resetujemy do domyślnej konfiguracji
    _config = DEFAULT_CONFIG.copy()

    # Ładujemy konfigurację z pliku jeśli podano ścieżkę
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                _config = _merge_configs(_config, file_config)
        except Exception as e:
            print(f"Błąd podczas ładowania konfiguracji z pliku: {e}")

    # Nadpisujemy wartości ze zmiennych środowiskowych
    env_config = {}

    # API
    if os.environ.get('AFE_API_HOST'):
        env_config.setdefault('api', {})['host'] = os.environ.get('AFE_API_HOST')
    if os.environ.get('AFE_API_PORT'):
        env_config.setdefault('api', {})['port'] = int(os.environ.get('AFE_API_PORT', '8000'))
    if os.environ.get('AFE_API_DEBUG'):
        env_config.setdefault('api', {})['debug'] = os.environ.get('AFE_API_DEBUG').lower() == 'true'
    if os.environ.get('AFE_API_KEY_REQUIRED'):
        env_config.setdefault('api', {})['api_key_required'] = os.environ.get('AFE_API_KEY_REQUIRED').lower() == 'true'

    # Integracje
    if os.environ.get('AFE_HUME_AI_API_KEY'):
        env_config.setdefault('integrations', {}).setdefault('hume_ai', {})['api_key'] = os.environ.get('AFE_HUME_AI_API_KEY')
        env_config.setdefault('integrations', {}).setdefault('hume_ai', {})['enabled'] = True

    if os.environ.get('AFE_DEEP_EVAL_API_KEY'):
        env_config.setdefault('integrations', {}).setdefault('deep_eval', {})['api_key'] = os.environ.get('AFE_DEEP_EVAL_API_KEY')
        env_config.setdefault('integrations', {}).setdefault('deep_eval', {})['enabled'] = True

    # Aktualizujemy konfigurację
    _config = _merge_configs(_config, env_config)

    # Tworzymy wymagane katalogi
    os.makedirs(Path(_config['storage_dir']), exist_ok=True)

    return _config

def get_config() -> Dict[str, Any]:
    """Zwraca aktualną konfigurację."""
    global _config
    return _config

def set_config_value(key_path: str, value: Any) -> None:
    """Ustawia wartość konfiguracji według podanej ścieżki kluczy.

    Args:
        key_path: Ścieżka kluczy oddzielonych kropkami, np. "api.host"
        value: Wartość do ustawienia
    """
    global _config

    keys = key_path.split('.')
    current = _config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value

# Inicjalizacja konfiguracji
load_config()
