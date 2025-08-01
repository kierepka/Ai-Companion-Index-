#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kompozytowy ewaluator przyjazności AI

Ta klasa łączy wiele różnych ewaluatorów w jeden, aby zapewnić
kompleksową ocenę przyjazności AI z różnych perspektyw.
"""

from typing import Dict, Any, List, Optional

from .base import BaseEvaluator
from ..utils.logging import setup_logger

logger = setup_logger("composite_evaluator")

class CompositeEvaluator(BaseEvaluator):
    """Kompozytowy ewaluator przyjazności AI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje ewaluator z konfiguracją.

        Args:
            config: Słownik z parametrami konfiguracyjnymi ewaluatora
        """
        super().__init__(config)

        # Domyślna konfiguracja
        default_config = {
            "evaluators": {},  # Słownik ewaluatorów do użycia
            "weights": {},    # Wagi dla każdego ewaluatora
            "threshold": 70.0  # Próg dla satysfakcjonującego wyniku na skali 0-100
        }

        # Łączymy domyślną konfigurację z dostarczoną
        if self.config:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
        else:
            self.config = default_config

        # Inicjalizujemy ewaluatory
        self.evaluators = {}

    def add_evaluator(self, name: str, evaluator: BaseEvaluator, weight: float = 1.0) -> None:
        """Dodaje ewaluator do kompozytu.

        Args:
            name: Nazwa ewaluatora
            evaluator: Instancja ewaluatora
            weight: Waga ewaluatora w ocenie końcowej
        """
        self.evaluators[name] = evaluator
        self.config["weights"][name] = weight
        logger.info(f"Dodano ewaluator '{name}' z wagą {weight}")

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ocenia przyjazność AI przy użyciu wszystkich ewaluatorów.

        Args:
            data: Słownik zawierający dane do oceny dla wszystkich ewaluatorów

        Returns:
            Słownik z wynikami oceny
        """
        # Sprawdzamy, czy mamy jakiekolwiek ewaluatory
        if not self.evaluators:
            logger.error("Brak zarejestrowanych ewaluatorów")
            return {"error": "Brak zarejestrowanych ewaluatorów"}

        # Wyniki dla każdego ewaluatora
        evaluator_results = {}

        # Oceniamy przy użyciu każdego ewaluatora
        for name, evaluator in self.evaluators.items():
            # Sprawdzamy, czy mamy dane dla tego ewaluatora
            evaluator_data = data.get(name, {})

            try:
                result = evaluator.evaluate(evaluator_data)
                evaluator_results[name] = result
                logger.info(f"Ewaluator '{name}' zakończył ocenę. Wynik: {evaluator.get_score():.2f}")
            except Exception as e:
                logger.error(f"Błąd podczas oceny przy użyciu ewaluatora '{name}': {e}")
                evaluator_results[name] = {"error": str(e)}

        # Obliczamy ważoną średnią wyników
        weights = self.config.get("weights", {})
        total_weight = 0.0
        weighted_sum = 0.0

        for name, result in evaluator_results.items():
            # Jeśli wynik zawiera błąd, pomijamy go
            if "error" in result:
                continue

            # Pobieramy wynik liczbowy
            score = result.get("score", 0.0)
            weight = weights.get(name, 1.0)

            total_weight += weight
            weighted_sum += score * weight

        # Normalizujemy wynik
        if total_weight > 0:
            normalized_score = weighted_sum / total_weight
        else:
            normalized_score = 0.0

        # Sprawdzamy, czy wynik jest satysfakcjonujący
        threshold = self.config.get("threshold", 70.0)
        is_satisfactory = normalized_score >= threshold

        # Zbieramy wszystkie rekomendacje
        all_recommendations = []
        for name, evaluator in self.evaluators.items():
            recommendations = evaluator.get_recommendations()
            if recommendations and recommendations != ["Brak dostępnych rekomendacji."]:
                all_recommendations.extend(recommendations)

        # Usuwamy duplikaty rekomendacji
        unique_recommendations = list(dict.fromkeys(all_recommendations))

        # Zapisujemy wyniki
        self.results = {
            "score": normalized_score,
            "threshold": threshold,
            "is_satisfactory": is_satisfactory,
            "evaluator_results": evaluator_results,
            "recommendations": unique_recommendations
        }

        logger.info(f"Ocena kompozytowa zakończona. Wynik: {normalized_score:.2f}/100")

        return self.results
