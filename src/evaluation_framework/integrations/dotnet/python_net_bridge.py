#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Most integracyjny Python.NET dla integracji .NET

Ten moduł zapewnia most między Pythonem a .NET,
umożliwiając bezpośrednie wywołanie kodu Python z .NET.

Wymaga zainstalowania pakietu Python.NET w środowisku.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union, Callable

from ...utils.logging import setup_logger
from ... import models, evaluation

logger = setup_logger("integrations.dotnet.bridge")

class PythonNetBridge:
    """Most integracyjny Python.NET dla frameworka AI Friendliness Evaluator."""

    def __init__(self):
        """Inicjalizuje most Python.NET."""
        try:
            # Ta biblioteka jest opcjonalna i wymagana tylko w przypadku
            # korzystania z mostu Python.NET
            import clr
            self.clr_available = True
            logger.info("Pomyślnie zainicjalizowano most Python.NET")
        except ImportError:
            self.clr_available = False
            logger.warning("Python.NET nie jest dostępny. Zainstaluj pakiet 'pythonnet' dla pełnej integracji z .NET.")

    def is_available(self) -> bool:
        """Sprawdza, czy most Python.NET jest dostępny.

        Returns:
            True jeśli most jest dostępny, False w przeciwnym razie
        """
        return self.clr_available

    def register_models(self):
        """Rejestruje modele matematyczne w środowisku .NET."""
        if not self.clr_available:
            logger.error("Python.NET nie jest dostępny. Nie można zarejestrować modeli.")
            return

        try:
            # Tworzymy fabrykę modeli dostępną z .NET
            def create_linear_model(config_json=None):
                """Tworzy liniowy model równań różniczkowych."""
                config = None
                if config_json:
                    import json
                    config = json.loads(config_json)
                return models.LinearDifferentialModel(config)

            def create_nonlinear_model(config_json=None):
                """Tworzy nieliniowy model równań różniczkowych."""
                config = None
                if config_json:
                    import json
                    config = json.loads(config_json)
                return models.NonlinearDifferentialModel(config)

            def create_reservoir_model(config_json=None):
                """Tworzy model rezerwuarowy."""
                config = None
                if config_json:
                    import json
                    config = json.loads(config_json)
                return models.ReservoirModel(config)

            # Eksportujemy funkcje do .NET
            import Python.Runtime as pyr
            pyr.Runtime.Globals["create_linear_model"] = create_linear_model
            pyr.Runtime.Globals["create_nonlinear_model"] = create_nonlinear_model
            pyr.Runtime.Globals["create_reservoir_model"] = create_reservoir_model

            logger.info("Pomyślnie zarejestrowano modele matematyczne w środowisku .NET")
        except Exception as e:
            logger.error(f"Błąd podczas rejestrowania modeli: {e}")

    def register_evaluators(self):
        """Rejestruje ewaluatory przyjazności w środowisku .NET."""
        if not self.clr_available:
            logger.error("Python.NET nie jest dostępny. Nie można zarejestrować ewaluatorów.")
            return

        try:
            # Tworzymy fabrykę ewaluatorów dostępną z .NET
            def create_ratio_evaluator(config_json=None):
                """Tworzy ewaluator proporcji 5:1."""
                config = None
                if config_json:
                    import json
                    config = json.loads(config_json)
                return evaluation.RatioEvaluator(config)

            def create_rubric_evaluator(config_json=None):
                """Tworzy ewaluator rubrykowy."""
                config = None
                if config_json:
                    import json
                    config = json.loads(config_json)
                return evaluation.RubricEvaluator(config)

            def create_composite_evaluator(config_json=None):
                """Tworzy ewaluator kompozytowy."""
                config = None
                if config_json:
                    import json
                    config = json.loads(config_json)
                return evaluation.CompositeEvaluator(config)

            # Eksportujemy funkcje do .NET
            import Python.Runtime as pyr
            pyr.Runtime.Globals["create_ratio_evaluator"] = create_ratio_evaluator
            pyr.Runtime.Globals["create_rubric_evaluator"] = create_rubric_evaluator
            pyr.Runtime.Globals["create_composite_evaluator"] = create_composite_evaluator

            logger.info("Pomyślnie zarejestrowano ewaluatory przyjazności w środowisku .NET")
        except Exception as e:
            logger.error(f"Błąd podczas rejestrowania ewaluatorów: {e}")

    def setup_python_path(self):
        """Konfiguruje ścieżkę Pythona dla Python.NET."""
        if not self.clr_available:
            return

        try:
            # Dodajemy bieżący katalog do ścieżki Pythona
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

            if parent_dir not in sys.path:
                sys.path.append(parent_dir)

            # Konfigurujemy ścieżkę w Python.NET
            import Python.Runtime as pyr
            pyr.Runtime.PythonPath = ';'.join(sys.path)

            logger.info(f"Skonfigurowano ścieżkę Pythona dla Python.NET: {pyr.Runtime.PythonPath}")
        except Exception as e:
            logger.error(f"Błąd podczas konfigurowania ścieżki Pythona: {e}")
