#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integracja z DeepEval dla oceny jakości konwersacji

Ten moduł zapewnia integrację z biblioteką DeepEval do oceny
jakości konwersacji AI.
"""

import requests
from typing import Dict, Any, List, Optional, Union

from ..config import get_config
from ..utils.logging import setup_logger

logger = setup_logger("integrations.deep_eval")

class DeepEvalClient:
    """Klient do integracji z API DeepEval."""

    def __init__(self, api_key: Optional[str] = None):
        """Inicjalizuje klienta DeepEval.

        Args:
            api_key: Klucz API DeepEval. Jeśli nie podano, zostanie użyty z konfiguracji.
        """
        config = get_config()
        self.api_key = api_key or config.get("integrations", {}).get("deep_eval", {}).get("api_key", "")

        if not self.api_key:
            logger.warning("Brak klucza API DeepEval. Niektóre funkcje mogą być niedostępne.")

        self.base_url = "https://api.deepeval.com/v1"

    def evaluate_conversation(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ocenia jakość konwersacji.

        Args:
            conversation: Lista słowników z wymianami w konwersacji,
                         każdy słownik powinien mieć klucze 'user' i 'assistant'

        Returns:
            Słownik z wynikami oceny konwersacji
        """
        if not self.api_key:
            logger.error("Brak klucza API DeepEval. Nie można przeprowadzić oceny.")
            return {"error": "Brak klucza API DeepEval"}

        try:
            url = f"{self.base_url}/conversation/evaluate"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"conversation": conversation}

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas oceny konwersacji: {e}")
            return {"error": str(e)}

    def evaluate_response(self, user_input: str, ai_response: str) -> Dict[str, Any]:
        """Ocenia jakość pojedynczej odpowiedzi AI.

        Args:
            user_input: Wypowiedź użytkownika
            ai_response: Odpowiedź AI

        Returns:
            Słownik z wynikami oceny odpowiedzi
        """
        if not self.api_key:
            logger.error("Brak klucza API DeepEval. Nie można przeprowadzić oceny.")
            return {"error": "Brak klucza API DeepEval"}

        try:
            url = f"{self.base_url}/response/evaluate"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "user_input": user_input,
                "ai_response": ai_response
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas oceny odpowiedzi: {e}")
            return {"error": str(e)}

    def evaluate_empathy(self, user_input: str, ai_response: str) -> Dict[str, Any]:
        """Ocenia poziom empatii w odpowiedzi AI.

        Args:
            user_input: Wypowiedź użytkownika
            ai_response: Odpowiedź AI

        Returns:
            Słownik z wynikami oceny empatii
        """
        if not self.api_key:
            logger.error("Brak klucza API DeepEval. Nie można przeprowadzić oceny.")
            return {"error": "Brak klucza API DeepEval"}

        try:
            url = f"{self.base_url}/metrics/empathy"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "user_input": user_input,
                "ai_response": ai_response
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas oceny empatii: {e}")
            return {"error": str(e)}

    def evaluate_consistency(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """Ocenia spójność odpowiedzi AI w konwersacji.

        Args:
            conversation: Lista słowników z wymianami w konwersacji,
                         każdy słownik powinien mieć klucze 'user' i 'assistant'

        Returns:
            Słownik z wynikami oceny spójności
        """
        if not self.api_key:
            logger.error("Brak klucza API DeepEval. Nie można przeprowadzić oceny.")
            return {"error": "Brak klucza API DeepEval"}

        try:
            url = f"{self.base_url}/metrics/consistency"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {"conversation": conversation}

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas oceny spójności: {e}")
            return {"error": str(e)}

    def evaluate_multiple_metrics(self, conversation: List[Dict[str, str]], 
                                 metrics: List[str]) -> Dict[str, Any]:
        """Ocenia konwersację według wielu metryk.

        Args:
            conversation: Lista słowników z wymianami w konwersacji
            metrics: Lista nazw metryk do oceny (np. ['empathy', 'consistency', 'relevance'])

        Returns:
            Słownik z wynikami oceny dla każdej metryki
        """
        if not self.api_key:
            logger.error("Brak klucza API DeepEval. Nie można przeprowadzić oceny.")
            return {"error": "Brak klucza API DeepEval"}

        try:
            url = f"{self.base_url}/metrics/multiple"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "conversation": conversation,
                "metrics": metrics
            }

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas oceny wielu metryk: {e}")
            return {"error": str(e)}
