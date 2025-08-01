#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trasy API dla oceny przyjazności AI
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from ...evaluation import RatioEvaluator, RubricEvaluator, CompositeEvaluator
from ...utils.logging import setup_logger

logger = setup_logger("api.routes.evaluation")

router = APIRouter()

# Modele danych
class DialogueExchange(BaseModel):
    """Model wymiany w dialogu."""
    user_input: str = Field(..., description="Wypowiedź użytkownika")
    ai_response: str = Field(..., description="Odpowiedź AI")
    expected_response_types: Optional[List[str]] = Field(None, description="Oczekiwane typy odpowiedzi")

class RatioEvaluationRequest(BaseModel):
    """Model żądania oceny proporcji 5:1."""
    dialogues: List[DialogueExchange] = Field(..., description="Lista wymian w dialogu")
    threshold: Optional[float] = Field(5.0, description="Próg dla satysfakcjonującego wyniku")

class RubricRating(BaseModel):
    """Model oceny dla pojedynczej rubryki."""
    emotion_recognition: Optional[float] = Field(None, ge=0.0, le=5.0, description="Ocena rozpoznawania emocji (0-5)")
    empathic_response: Optional[float] = Field(None, ge=0.0, le=5.0, description="Ocena odpowiedzi empatycznej (0-5)")
    consistency: Optional[float] = Field(None, ge=0.0, le=5.0, description="Ocena spójności (0-5)")
    personalization: Optional[float] = Field(None, ge=0.0, le=5.0, description="Ocena personalizacji (0-5)")
    ethical_alignment: Optional[float] = Field(None, ge=0.0, le=5.0, description="Ocena zgodności etycznej (0-5)")

class RubricEvaluationRequest(BaseModel):
    """Model żądania oceny rubrykowej."""
    ratings: Dict[str, float] = Field(..., description="Oceny dla każdej rubryki")
    threshold: Optional[float] = Field(70.0, description="Próg dla satysfakcjonującego wyniku (0-100)")

class CompositeEvaluationRequest(BaseModel):
    """Model żądania oceny kompozytowej."""
    ratio_evaluation: Optional[RatioEvaluationRequest] = Field(None, description="Dane do oceny proporcji 5:1")
    rubric_evaluation: Optional[RubricEvaluationRequest] = Field(None, description="Dane do oceny rubrykowej")
    threshold: Optional[float] = Field(70.0, description="Próg dla satysfakcjonującego wyniku (0-100)")

# Trasy API
@router.post("/ratio", status_code=status.HTTP_200_OK)
async def evaluate_ratio(request: RatioEvaluationRequest):
    """Ocenia przyjazność AI na podstawie proporcji 5:1."""
    try:
        # Konwertujemy dane z modelu Pydantic do słownika
        data = {"dialogues": []}
        for dialogue in request.dialogues:
            dialogue_dict = {
                "user_input": dialogue.user_input,
                "ai_response": dialogue.ai_response
            }
            if dialogue.expected_response_types:
                dialogue_dict["expected_response_types"] = dialogue.expected_response_types
            data["dialogues"].append(dialogue_dict)

        # Tworzymy i konfigurujemy ewaluator
        evaluator = RatioEvaluator({"threshold": request.threshold})

        # Przeprowadzamy ocenę
        result = evaluator.evaluate(data)

        return result
    except Exception as e:
        logger.error(f"Błąd podczas oceny proporcji 5:1: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas oceny: {str(e)}"
        )

@router.post("/rubric", status_code=status.HTTP_200_OK)
async def evaluate_rubric(request: RubricEvaluationRequest):
    """Ocenia przyjazność AI na podstawie rubryk."""
    try:
        # Konwertujemy dane z modelu Pydantic do słownika
        data = {"ratings": request.ratings}

        # Tworzymy i konfigurujemy ewaluator
        evaluator = RubricEvaluator({"threshold": request.threshold})

        # Przeprowadzamy ocenę
        result = evaluator.evaluate(data)

        return result
    except Exception as e:
        logger.error(f"Błąd podczas oceny rubrykowej: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas oceny: {str(e)}"
        )

@router.post("/composite", status_code=status.HTTP_200_OK)
async def evaluate_composite(request: CompositeEvaluationRequest):
    """Przeprowadza kompozytową ocenę przyjazności AI."""
    try:
        # Tworzymy ewaluator kompozytowy
        evaluator = CompositeEvaluator({"threshold": request.threshold})

        # Dodajemy ewaluatory składowe, jeśli są dostępne dane
        data = {}

        if request.ratio_evaluation:
            ratio_evaluator = RatioEvaluator()
            evaluator.add_evaluator("ratio", ratio_evaluator, weight=0.6)

            # Przygotowujemy dane dla ewaluatora proporcji
            ratio_data = {"dialogues": []}
            for dialogue in request.ratio_evaluation.dialogues:
                dialogue_dict = {
                    "user_input": dialogue.user_input,
                    "ai_response": dialogue.ai_response
                }
                if dialogue.expected_response_types:
                    dialogue_dict["expected_response_types"] = dialogue.expected_response_types
                ratio_data["dialogues"].append(dialogue_dict)

            data["ratio"] = ratio_data

        if request.rubric_evaluation:
            rubric_evaluator = RubricEvaluator()
            evaluator.add_evaluator("rubric", rubric_evaluator, weight=0.4)

            # Przygotowujemy dane dla ewaluatora rubrykowego
            rubric_data = {"ratings": request.rubric_evaluation.ratings}
            data["rubric"] = rubric_data

        # Przeprowadzamy ocenę
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Brak danych do oceny. Wymagane są dane dla co najmniej jednego ewaluatora."
            )

        result = evaluator.evaluate(data)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas oceny kompozytowej: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas oceny: {str(e)}"
        )
