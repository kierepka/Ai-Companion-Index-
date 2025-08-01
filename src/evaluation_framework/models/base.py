#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bazowa klasa dla modeli dynamiki relacji AI-użytkownik

Ta klasa definiuje wspólny interfejs dla wszystkich modeli matematycznych
używanych do modelowania dynamiki relacji między AI a użytkownikiem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

class BaseRelationshipModel(ABC):
    """Bazowa klasa abstrakcyjna dla modeli dynamiki relacji."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje model z konfiguracją.

        Args:
            config: Słownik z parametrami konfiguracyjnymi modelu
        """
        self.config = config or {}
        self.parameters = {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Dopasowuje model do danych.

        Args:
            data: Słownik zawierający dane, co najmniej:
                  - 't': wektor czasu
                  - 'H': wektor przywiązania użytkownika
                  - 'A': wektor zaangażowania AI

        Returns:
            Słownik z wynikami dopasowania, zawierający co najmniej:
            - 'parameters': dopasowane parametry
            - 'error': błąd dopasowania
        """
        pass

    @abstractmethod
    def predict(self, t: np.ndarray, initial_conditions: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Przewiduje trajektorię modelu dla podanych czasów i warunków początkowych.

        Args:
            t: Wektor czasów do przewidywania
            initial_conditions: Słownik z warunkami początkowymi

        Returns:
            Słownik z przewidywanymi trajektoriami dla każdej zmiennej stanu
        """
        pass

    @abstractmethod
    def analyze_stability(self) -> Dict[str, Any]:
        """Analizuje stabilność modelu na podstawie jego parametrów.

        Returns:
            Słownik z wynikami analizy stabilności
        """
        pass

    @abstractmethod
    def get_equilibrium_points(self) -> List[Dict[str, float]]:
        """Oblicza punkty równowagi modelu.

        Returns:
            Lista słowników zawierających współrzędne punktów równowagi
        """
        pass

    def get_parameters(self) -> Dict[str, Any]:
        """Zwraca parametry modelu.

        Returns:
            Słownik z parametrami modelu
        """
        return self.parameters

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Ustawia parametry modelu.

        Args:
            parameters: Słownik z parametrami modelu
        """
        self.parameters = parameters
        self.is_fitted = True

    def get_parameter_explanations(self) -> Dict[str, str]:
        """Generuje wyjaśnienia znaczenia parametrów modelu.

        Returns:
            Słownik z wyjaśnieniami dla każdego parametru
        """
        if not self.is_fitted:
            return {"error": "Model nie został jeszcze dopasowany."}

        return self._generate_parameter_explanations()

    @abstractmethod
    def _generate_parameter_explanations(self) -> Dict[str, str]:
        """Implementacja generowania wyjaśnień parametrów dla konkretnego modelu."""
        pass
