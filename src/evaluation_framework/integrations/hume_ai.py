#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integracja z API Hume AI dla rozpoznawania emocji

Ten moduł zapewnia integrację z API Hume AI do analizy emocji
w tekście i mowie.
"""

import requests
from typing import Dict, Any, List, Optional, Union

from ..config import get_config
from ..utils.logging import setup_logger

logger = setup_logger("integrations.hume_ai")

class HumeAIClient:
    """Klient do integracji z API Hume AI."""

    def __init__(self, api_key: Optional[str] = None):
        """Inicjalizuje klienta Hume AI.

        Args:
            api_key: Klucz API Hume AI. Jeśli nie podano, zostanie użyty z konfiguracji.
        """
        config = get_config()
        self.api_key = api_key or config.get("integrations", {}).get("hume_ai", {}).get("api_key", "")

        if not self.api_key:
            logger.warning("Brak klucza API Hume AI. Niektóre funkcje mogą być niedostępne.")

        self.base_url = "https://api.hume.ai/v0"

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analizuje emocje w tekście.

        Args:
            text: Tekst do analizy

        Returns:
            Słownik z wynikami analizy emocji
        """
        if not self.api_key:
            logger.error("Brak klucza API Hume AI. Nie można przeprowadzić analizy.")
            return {"error": "Brak klucza API Hume AI"}

        try:
            url = f"{self.base_url}/language/emotions"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {"text": text}

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas analizy tekstu: {e}")
            return {"error": str(e)}

    def analyze_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Analizuje emocje w wielu tekstach.

        Args:
            texts: Lista tekstów do analizy

        Returns:
            Słownik z wynikami analizy emocji dla każdego tekstu
        """
        if not self.api_key:
            logger.error("Brak klucza API Hume AI. Nie można przeprowadzić analizy.")
            return {"error": "Brak klucza API Hume AI"}

        try:
            url = f"{self.base_url}/language/emotions"
            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            payload = {"texts": texts}

            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Błąd podczas analizy wielu tekstów: {e}")
            return {"error": str(e)}

    def get_dominant_emotions(self, text: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """Zwraca dominujące emocje w tekście.

        Args:
            text: Tekst do analizy
            top_n: Liczba najsilniejszych emocji do zwrócenia

        Returns:
            Lista słowników z najsilniejszymi emocjami i ich wartościami
        """
        results = self.analyze_text(text)

        if "error" in results:
            return []

        try:
            # Parsujemy wyniki i wyodrębniamy emocje
            emotions = results.get("emotions", [])

            # Sortujemy emocje według ich wartości
            sorted_emotions = sorted(emotions, key=lambda x: x.get("score", 0), reverse=True)

            # Zwracamy top_n emocji
            return sorted_emotions[:top_n]
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania wyników analizy: {e}")
            return []

    def match_expected_emotions(self, text: str, expected_emotions: List[str]) -> Dict[str, Any]:
        """Sprawdza, czy tekst zawiera oczekiwane emocje.

        Args:
            text: Tekst do analizy
            expected_emotions: Lista oczekiwanych emocji

        Returns:
            Słownik z wynikami dopasowania emocji
        """
        results = self.analyze_text(text)

        if "error" in results:
            return {"error": results["error"], "matches": [], "match_rate": 0.0}

        try:
            # Parsujemy wyniki i wyodrębniamy emocje
            detected_emotions = results.get("emotions", [])

            # Konwertujemy na słownik dla łatwiejszego dostępu
            emotion_dict = {emotion["name"].lower(): emotion["score"] for emotion in detected_emotions}

            # Sprawdzamy dopasowanie dla każdej oczekiwanej emocji
            matches = []
            for expected in expected_emotions:
                expected_lower = expected.lower()
                if expected_lower in emotion_dict:
                    matches.append({
                        "emotion": expected,
                        "detected": True,
                        "score": emotion_dict[expected_lower]
                    })
                else:
                    matches.append({
                        "emotion": expected,
                        "detected": False,
                        "score": 0.0
                    })

            # Obliczamy wskaźnik dopasowania
            matched_count = sum(1 for m in matches if m["detected"])
            match_rate = matched_count / len(expected_emotions) if expected_emotions else 0.0

            return {
                "matches": matches,
                "match_rate": match_rate,
                "detected_emotions": detected_emotions
            }
        except Exception as e:
            logger.error(f"Błąd podczas dopasowywania emocji: {e}")
            return {"error": str(e), "matches": [], "match_rate": 0.0}
