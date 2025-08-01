#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ewaluator rubrykowy dla oceny przyjazności AI

Ta klasa implementuje metodę oceny opartą na rubrykach, które określają
kryteria oceny dla różnych wymiarów przyjazności AI.
"""

import os
from pathlib import Path
import re
from typing import Dict, Any, List, Optional

from .base import BaseEvaluator
from ..utils.logging import setup_logger

logger = setup_logger("rubric_evaluator")

class RubricEvaluator(BaseEvaluator):
    """Ewaluator rubrykowy dla oceny przyjazności AI."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Inicjalizuje ewaluator z konfiguracją.

        Args:
            config: Słownik z parametrami konfiguracyjnymi ewaluatora
        """
        super().__init__(config)

        # Domyślna konfiguracja
        default_config = {
            "rubrics": [
                "emotion_recognition",
                "empathic_response",
                "consistency",
                "personalization",
                "ethical_alignment"
            ],
            "rubrics_dir": "./rubrics",
            "weights": {
                "emotion_recognition": 0.2,
                "empathic_response": 0.3,
                "consistency": 0.15,
                "personalization": 0.2,
                "ethical_alignment": 0.15
            },
            "threshold": 70.0  # Próg dla satysfakcjonującego wyniku na skali 0-100
        }

        # Łączymy domyślną konfigurację z dostarczoną
        if self.config:
            for key, value in default_config.items():
                if key not in self.config:
                    self.config[key] = value
                elif key == "weights" and "weights" in self.config:
                    # Łączymy słowniki wag
                    for rubric, weight in value.items():
                        if rubric not in self.config["weights"]:
                            self.config["weights"][rubric] = weight
        else:
            self.config = default_config

        # Ładujemy rubryki
        self.rubrics = self._load_rubrics()

    def _load_rubrics(self) -> Dict[str, Dict[str, Any]]:
        """Ładuje rubryki z plików.

        Returns:
            Słownik zawierający załadowane rubryki
        """
        rubrics = {}
        rubrics_dir = Path(self.config.get('rubrics_dir', './rubrics'))

        for rubric_name in self.config.get('rubrics', []):
            rubric_path = rubrics_dir / f"{rubric_name}.md"

            if not rubric_path.exists():
                logger.warning(f"Nie znaleziono rubryki: {rubric_path}")
                continue

            try:
                with open(rubric_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Parsujemy zawartość rubryki
                rubric = self._parse_rubric(content, rubric_name)
                rubrics[rubric_name] = rubric
                logger.info(f"Załadowano rubrykę: {rubric_name}")
            except Exception as e:
                logger.error(f"Błąd podczas ładowania rubryki {rubric_name}: {e}")

        return rubrics

    def _parse_rubric(self, content: str, rubric_name: str) -> Dict[str, Any]:
        """Parsuje zawartość pliku rubryki.

        Args:
            content: Zawartość pliku rubryki w formacie Markdown
            rubric_name: Nazwa rubryki

        Returns:
            Sparsowana rubryka jako słownik
        """
        # Wyodrębniamy tytuł
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else rubric_name

        # Wyodrębniamy opisy dla każdego poziomu (0-5)
        level_descriptions = {}
        level_matches = re.finditer(r'###\s+(\d+)\s*-\s*(.+?)\n(.*?)(?=###|$)', 
                                   content, re.DOTALL)

        for match in level_matches:
            level = int(match.group(1))
            level_title = match.group(2).strip()
            level_description = match.group(3).strip()

            level_descriptions[level] = {
                "title": level_title,
                "description": level_description
            }

        # Wyodrębniamy metodologię oceny
        methodology_match = re.search(r'##\s+Metodologia oceny\s*\n(.+?)(?=##|$)', 
                                     content, re.DOTALL)
        methodology = methodology_match.group(1).strip() if methodology_match else ""

        return {
            "title": title,
            "levels": level_descriptions,
            "methodology": methodology
        }

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ocenia przyjazność AI na podstawie dostarczonych danych.

        Args:
            data: Słownik zawierający dane do oceny, musi zawierać klucz 'ratings'
                 z ocenami dla każdej rubryki

        Returns:
            Słownik z wynikami oceny
        """
        # Sprawdzamy, czy mamy potrzebne dane
        if 'ratings' not in data:
            logger.error("Brak wymaganych danych: 'ratings'")
            return {'error': "Brak wymaganych danych: 'ratings'"}

        ratings = data['ratings']

        # Sprawdzamy, czy mamy oceny dla wszystkich rubryk
        missing_rubrics = []
        for rubric_name in self.config.get('rubrics', []):
            if rubric_name not in ratings:
                missing_rubrics.append(rubric_name)

        if missing_rubrics:
            logger.warning(f"Brak ocen dla rubryk: {', '.join(missing_rubrics)}")

        # Obliczamy ważoną średnią ocen
        weights = self.config.get('weights', {})
        total_weight = 0.0
        weighted_sum = 0.0

        # Szczegóły oceny dla każdej rubryki
        rating_details = {}

        for rubric_name, rating in ratings.items():
            if rubric_name in weights:
                weight = weights[rubric_name]
                total_weight += weight
                weighted_sum += rating * weight

                # Dodajemy szczegóły tej oceny
                rating_details[rubric_name] = {
                    "score": rating,
                    "weight": weight,
                    "weighted_score": rating * weight
                }

                # Jeśli mamy rubrykę w systemie, dodajemy jej tytuł i opisy poziomów
                if rubric_name in self.rubrics:
                    rubric = self.rubrics[rubric_name]
                    rating_details[rubric_name]["title"] = rubric.get("title", rubric_name)

                    # Dodajemy opis najbliższego poziomu
                    nearest_level = round(rating)
                    if nearest_level in rubric.get("levels", {}):
                        level_info = rubric["levels"][nearest_level]
                        rating_details[rubric_name]["level_title"] = level_info.get("title", "")
                        rating_details[rubric_name]["level_description"] = level_info.get("description", "")

        # Normalizujemy wynik do skali 0-100
        if total_weight > 0:
            normalized_score = (weighted_sum / total_weight) * 20.0  # Skala 0-5 -> 0-100
        else:
            normalized_score = 0.0

        # Sprawdzamy, czy wynik jest satysfakcjonujący
        threshold = self.config.get('threshold', 70.0)
        is_satisfactory = normalized_score >= threshold

        # Generujemy rekomendacje
        recommendations = self._generate_recommendations(rating_details, normalized_score)

        # Zapisujemy wyniki
        self.results = {
            "score": normalized_score,
            "threshold": threshold,
            "is_satisfactory": is_satisfactory,
            "rating_details": rating_details,
            "recommendations": recommendations
        }

        logger.info(f"Ocena rubrykowa zakończona. Wynik: {normalized_score:.2f}/100")

        return self.results

    def _generate_recommendations(self, rating_details: Dict[str, Dict[str, Any]], 
                                 overall_score: float) -> List[str]:
        """Generuje rekomendacje na podstawie wyników oceny.

        Args:
            rating_details: Szczegóły oceny dla każdej rubryki
            overall_score: Ogólny wynik oceny

        Returns:
            Lista rekomendacji
        """
        recommendations = []
        threshold = self.config.get('threshold', 70.0)

        if overall_score < threshold:
            recommendations.append(f"Podnieś ogólny wynik (obecnie {overall_score:.2f}) do co najmniej {threshold}.")

            # Identyfikujemy rubryki z najniższymi wynikami
            sorted_ratings = sorted(rating_details.items(), 
                                   key=lambda x: x[1]['score'])

            # Dodajemy rekomendacje dla 2-3 najsłabszych rubryk
            for rubric_name, details in sorted_ratings[:min(3, len(sorted_ratings))]:
                score = details['score']
                if score < 4.0:  # Sugerujemy poprawę tylko dla ocen poniżej 4.0
                    title = details.get('title', rubric_name)
                    recommendations.append(f"Popraw wynik w rubryce '{title}' (obecnie {score:.2f}/5.0).")

                    # Jeśli mamy opis poziomu wyższego, dodajemy go jako wskazówkę
                    current_level = round(score)
                    next_level = current_level + 1

                    if rubric_name in self.rubrics and next_level in self.rubrics[rubric_name].get("levels", {}):
                        next_level_info = self.rubrics[rubric_name]["levels"][next_level]
                        next_level_title = next_level_info.get("title", "")
                        recommendations.append(f"  Dąż do osiągnięcia poziomu '{next_level_title}'.")
        else:
            # Jeśli wynik jest satysfakcjonujący, ale wciąż można go poprawić
            if overall_score < 90.0:
                recommendations.append(f"Dobry wynik ({overall_score:.2f}/100), ale wciąż można go poprawić.")

                # Identyfikujemy rubrykę z najniższym wynikiem
                weakest_rubric = min(rating_details.items(), 
                                    key=lambda x: x[1]['score'])
                rubric_name = weakest_rubric[0]
                score = weakest_rubric[1]['score']

                if score < 4.5:  # Sugerujemy poprawę tylko dla ocen poniżej 4.5
                    title = weakest_rubric[1].get('title', rubric_name)
                    recommendations.append(f"Skup się na poprawie rubryki '{title}' (obecnie {score:.2f}/5.0).")
            else:
                recommendations.append("Doskonały wynik! Utrzymuj ten poziom przyjazności.")

        return recommendations
