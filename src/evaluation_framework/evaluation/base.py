#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bazowa klasa dla ewaluatorów przyjazności AI

Ta klasa definiuje wspólny interfejs dla wszystkich ewaluatorów
używanych do oceny przyjazności AI wobec użytkownika.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseEvaluator(ABC):
    """Bazowa klasa abstrakcyjna dla ewaluatorów przyjazności AI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje ewaluator z konfiguracją.

        Args:
            config: Słownik z parametrami konfiguracyjnymi ewaluatora
        """
        self.config = config or {}
        self.results = {}

    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ocenia przyjazność AI na podstawie dostarczonych danych.

        Args:
            data: Słownik zawierający dane do oceny, specyficzne dla danego ewaluatora

        Returns:
            Słownik z wynikami oceny
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """Zwraca wyniki ostatniej oceny.

        Returns:
            Słownik z wynikami oceny
        """
        return self.results

    def get_score(self) -> float:
        """Zwraca ogólny wynik liczbowy z ostatniej oceny.

        Returns:
            Wynik oceny jako liczba zmiennoprzecinkowa
        """
        if not self.results:
            return 0.0

        if 'score' in self.results:
            return self.results['score']
        else:
            return 0.0

    def is_satisfactory(self) -> bool:
        """Sprawdza, czy wynik oceny jest satysfakcjonujący.

        Returns:
            True jeśli wynik jest satysfakcjonujący, False w przeciwnym razie
        """
        if not self.results:
            return False

        if 'is_satisfactory' in self.results:
            return self.results['is_satisfactory']
        elif 'score' in self.results and 'threshold' in self.results:
            return self.results['score'] >= self.results['threshold']
        else:
            return False

    def get_recommendations(self) -> List[str]:
        """Zwraca rekomendacje na podstawie wyników oceny.

        Returns:
            Lista rekomendacji jako stringi
        """
        if not self.results or 'recommendations' not in self.results:
            return ["Brak dostępnych rekomendacji."]

        return self.results['recommendations']

    def set_config(self, config: Dict[str, Any]) -> None:
        """Ustawia konfigurację ewaluatora.

        Args:
            config: Słownik z parametrami konfiguracyjnymi
        """
        self.config = config

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Aktualizuje konfigurację ewaluatora.

        Args:
            updates: Słownik z parametrami do zaktualizowania
        """
        self.config.update(updates)
