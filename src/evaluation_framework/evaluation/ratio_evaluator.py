#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ewaluator proporcji 5:1 dla oceny przyjazności AI

Ta klasa implementuje metodę oceny opartą na regule 5:1 inspirowanej
pracami Johna Gottmana, która analizuje stosunek pozytywnych do
negatywnych interakcji w relacji.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

from .base import BaseEvaluator
from ..utils.logging import setup_logger

logger = setup_logger("ratio_evaluator")

class RatioEvaluator(BaseEvaluator):
    """Ewaluator proporcji 5:1 dla oceny przyjazności AI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje ewaluator z konfiguracją.

        Args:
            config: Słownik z parametrami konfiguracyjnymi ewaluatora
        """
        super().__init__(config)

        # Domyślna konfiguracja
        default_config = {
            "threshold": 5.0,  # Próg dla satysfakcjonującego wyniku
            "positive_types": [
                "empathic", "supportive", "personalized", "affirming", "validating",
                "reflective", "celebratory", "encouraging", "normalizing", "gentle_guidance",
                "acknowledging", "resource_offering", "non_judgmental"
            ],
            "negative_types": [
                "ignoring_emotion", "changing_subject", "minimizing_feelings", 
                "generic_platitudes", "toxic_positivity", "contradicting_previous_knowledge",
                "asking_already_provided_information", "generic_responses_ignoring_history",
                "minimizing_severity", "philosophical_debate"
            ],
            "interaction_patterns": {
                "empathic": r"rozumiem|przykro mi|współczuję|musi być ci trudno",
                "supportive": r"jestem (tu|tutaj) dla ciebie|mogę pomóc|wsparcie|wspierać",
                "validating": r"to normalne|naturaln(e|a)|zrozumiał(e|a)",
                "reflective": r"wydaje się, że czujesz|słyszę, że|wygląda na to, że",
                "normalizing": r"wiel(e|u) osób|często|typow(e|a)|powszechn(e|a)",
                "ignoring_emotion": r"^(Ale|Jednak|Zmieńmy temat)",
                "changing_subject": r"wracając do|przejdźmy do|zmieńmy temat|a tak swoją drogą",
                "minimizing_feelings": r"nie jest tak źle|przesadzasz|to nic takiego|nie ma co się martwić",
                "toxic_positivity": r"zawsze myśl pozytywnie|po prostu bądź szczęśliw|wszystko będzie dobrze bez",
                "generic_platitudes": r"czas leczy rany|wszystko będzie dobrze|jutro będzie lepiej"
            },
            "generic_response_patterns": [
                r"^Rozumiem\.$",
                r"^Dziękuję za informację\.$",
                r"^Mogę jakoś pomóc\?$",
                r"^To interesujące\.$"
            ],
            "generic_min_length": 20  # Minimalna długość odpowiedzi, poniżej której uznajemy ją za generyczną
        }

        # Łączymy domyślną konfigurację z dostarczoną
        if self.config:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
        else:
            self.config = default_config

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ocenia przyjazność AI na podstawie dostarczonych danych.

        Args:
            data: Słownik zawierający dane do oceny, musi zawierać klucz 'dialogues'
                 z listą wymian, gdzie każda wymiana ma 'user_input' i 'ai_response'

        Returns:
            Słownik z wynikami oceny
        """
        # Sprawdzamy, czy mamy wszystkie potrzebne dane
        if 'dialogues' not in data:
            logger.error("Brak wymaganych danych: 'dialogues'")
            return {'error': "Brak wymaganych danych: 'dialogues'"}

        dialogues = data['dialogues']

        # Inicjalizujemy liczniki
        total_positive = 0
        total_negative = 0
        evaluation_details = []

        # Analizujemy każdą wymianę
        for dialogue in dialogues:
            if 'user_input' not in dialogue or 'ai_response' not in dialogue:
                logger.warning("Pominięto niepełną wymianę: brak user_input lub ai_response")
                continue

            user_input = dialogue['user_input']
            ai_response = dialogue['ai_response']
            expected_types = dialogue.get('expected_response_types', [])

            # Klasyfikujemy interakcję
            pos_count, neg_count, pos_types, neg_types = self._classify_interaction(
                user_input, ai_response, expected_types)

            total_positive += pos_count
            total_negative += neg_count

            # Zapisujemy szczegóły oceny dla tej wymiany
            evaluation_details.append({
                "user_input": user_input,
                "ai_response": ai_response,
                "positive_count": pos_count,
                "negative_count": neg_count,
                "detected_positive_types": pos_types,
                "detected_negative_types": neg_types
            })

        # Obliczamy wskaźnik przyjazności (Friendliness Index)
        if total_negative == 0:
            # Aby uniknąć dzielenia przez zero, jeśli nie ma negatywnych interakcji,
            # ustawiamy wskaźnik na wysoki, ale skończony
            friendliness_index = 10.0
        else:
            friendliness_index = total_positive / total_negative

        # Określamy, czy wynik jest satysfakcjonujący
        threshold = self.config.get('threshold', 5.0)
        is_satisfactory = friendliness_index >= threshold

        # Generujemy rekomendacje
        recommendations = self._generate_recommendations(
            total_positive, total_negative, friendliness_index, evaluation_details)

        # Zapisujemy wyniki
        self.results = {
            "total_positive_interactions": total_positive,
            "total_negative_interactions": total_negative,
            "friendliness_index": friendliness_index,
            "threshold": threshold,
            "is_satisfactory": is_satisfactory,
            "score": min(10.0, friendliness_index) / 10.0 * 100.0,  # Konwersja na skalę 0-100
            "details": evaluation_details,
            "recommendations": recommendations
        }

        logger.info(f"Ocena proporcji 5:1 zakończona. Indeks przyjazności: {friendliness_index:.2f}")

        return self.results

    def _classify_interaction(self, user_input: str, ai_response: str, 
                             expected_types: List[str]) -> Tuple[int, int, List[str], List[str]]:
        """Klasyfikuje interakcję jako pozytywną lub negatywną.

        Args:
            user_input: Wypowiedź użytkownika
            ai_response: Odpowiedź AI
            expected_types: Oczekiwane typy odpowiedzi

        Returns:
            Krotka (liczba_pozytywnych, liczba_negatywnych, typy_pozytywne, typy_negatywne)
        """
        positive_count = 0
        negative_count = 0
        detected_positives = []
        detected_negatives = []

        # Pobieramy konfigurację
        positive_types = self.config.get('positive_types', [])
        negative_types = self.config.get('negative_types', [])
        interaction_patterns = self.config.get('interaction_patterns', {})

        # Sprawdzamy pozytywne typy interakcji
        for pos_type in positive_types:
            if pos_type in expected_types and self._is_interaction_type_present(ai_response, pos_type, interaction_patterns):
                positive_count += 1
                detected_positives.append(pos_type)

        # Sprawdzamy negatywne typy interakcji
        for neg_type in negative_types:
            if self._is_interaction_type_present(ai_response, neg_type, interaction_patterns):
                negative_count += 1
                detected_negatives.append(neg_type)

        # Jeśli odpowiedź jest bardzo generyczna, liczymy to jako negatywną interakcję
        if self._is_generic_response(ai_response) and "generic_response" not in detected_negatives:
            negative_count += 1
            detected_negatives.append("generic_response")

        return positive_count, negative_count, detected_positives, detected_negatives

    def _is_interaction_type_present(self, response: str, interaction_type: str, 
                                    patterns: Dict[str, str]) -> bool:
        """Sprawdza, czy odpowiedź zawiera dany typ interakcji.

        Args:
            response: Odpowiedź AI do analizy
            interaction_type: Typ interakcji do sprawdzenia
            patterns: Słownik z wzorcami dla różnych typów interakcji

        Returns:
            True jeśli typ interakcji jest obecny, False w przeciwnym razie
        """
        pattern = patterns.get(interaction_type, "")
        if pattern:
            return bool(re.search(pattern, response.lower()))

        # Dla typów bez zdefiniowanego wzorca, zwracamy False
        return False

    def _is_generic_response(self, response: str) -> bool:
        """Sprawdza, czy odpowiedź jest generyczna.

        Args:
            response: Odpowiedź AI do analizy

        Returns:
            True jeśli odpowiedź jest generyczna, False w przeciwnym razie
        """
        # Sprawdzamy, czy odpowiedź jest bardzo krótka
        min_length = self.config.get('generic_min_length', 20)
        if len(response.strip()) < min_length:
            return True

        # Sprawdzamy wzorce generycznych odpowiedzi
        generic_patterns = self.config.get('generic_response_patterns', [])
        for pattern in generic_patterns:
            if re.search(pattern, response):
                return True

        return False

    def _generate_recommendations(self, total_positive: int, total_negative: int, 
                                 friendliness_index: float, 
                                 details: List[Dict[str, Any]]) -> List[str]:
        """Generuje rekomendacje na podstawie wyników oceny.

        Args:
            total_positive: Całkowita liczba pozytywnych interakcji
            total_negative: Całkowita liczba negatywnych interakcji
            friendliness_index: Wskaźnik przyjazności
            details: Szczegóły oceny dla każdej wymiany

        Returns:
            Lista rekomendacji
        """
        recommendations = []
        threshold = self.config.get('threshold', 5.0)

        if friendliness_index < threshold:
            recommendations.append(f"Zwiększ wskaźnik przyjazności (obecnie {friendliness_index:.2f}) do co najmniej {threshold}.")

            # Zbieramy wszystkie negatywne typy ze wszystkich interakcji
            all_negative_types = []
            for detail in details:
                all_negative_types.extend(detail['detected_negative_types'])

            # Liczymy częstotliwość występowania
            negative_counts = Counter(all_negative_types)

            # Dodajemy rekomendacje dla najczęstszych negatywnych typów
            for neg_type, count in negative_counts.most_common(3):
                if neg_type == "generic_response":
                    recommendations.append(f"Unikaj generycznych, krótkich odpowiedzi. Dodaj więcej kontekstu i personalizacji.")
                elif neg_type == "ignoring_emotion":
                    recommendations.append(f"Zwracaj większą uwagę na emocje użytkownika i odpowiednio je adresuj.")
                elif neg_type == "changing_subject":
                    recommendations.append(f"Unikaj zmiany tematu, kiedy użytkownik chce omówić konkretną kwestię.")
                elif neg_type == "minimizing_feelings":
                    recommendations.append(f"Nie umniejszaj uczuć użytkownika. Zamiast tego waliduj je.")
                elif neg_type == "toxic_positivity":
                    recommendations.append(f"Unikaj wymuszania pozytywnego nastawienia, gdy użytkownik wyraża trudne emocje.")
                elif neg_type == "generic_platitudes":
                    recommendations.append(f"Zastąp ogólnikowe frazy konkretnym, spersonalizowanym wsparciem.")
                else:
                    recommendations.append(f"Zredukuj występowanie typu interakcji '{neg_type}'.")
        else:
            # Jeśli wynik jest satysfakcjonujący, ale wciąż można go poprawić
            if total_negative > 0:
                recommendations.append(f"Utrzymuj wysoki wskaźnik przyjazności ({friendliness_index:.2f}).")

                # Zbieramy wszystkie negatywne typy
                all_negative_types = []
                for detail in details:
                    all_negative_types.extend(detail['detected_negative_types'])

                # Jeśli są jakieś negatywne interakcje, sugerujemy poprawę
                if all_negative_types:
                    neg_type = Counter(all_negative_types).most_common(1)[0][0]
                    recommendations.append(f"Wyeliminuj pozostałe negatywne interakcje typu '{neg_type}'.")
            else:
                recommendations.append("Doskonały wynik! Utrzymuj ten poziom przyjazności.")

            # Dodajemy sugestie dotyczące zwiększenia różnorodności pozytywnych interakcji
            all_positive_types = []
            for detail in details:
                all_positive_types.extend(detail['detected_positive_types'])

            positive_counts = Counter(all_positive_types)
            if len(positive_counts) < 3 and total_positive > 0:
                recommendations.append("Zwiększ różnorodność pozytywnych interakcji.")

        return recommendations
