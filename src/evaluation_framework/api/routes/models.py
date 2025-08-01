#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trasy API dla modeli dynamiki relacji
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import numpy as np

from ...models import LinearDifferentialModel, NonlinearDifferentialModel, ReservoirModel
from ...utils.logging import setup_logger

logger = setup_logger("api.routes.models")

router = APIRouter()

# Modele danych
class TimeSeriesData(BaseModel):
    """Model danych szeregu czasowego."""
    t: List[float] = Field(..., description="Wektor czasu")
    H: List[float] = Field(..., description="Wektor przywiązania użytkownika")
    A: List[float] = Field(..., description="Wektor zaangażowania AI")

class ModelFitRequest(BaseModel):
    """Model żądania dopasowania modelu."""
    data: TimeSeriesData = Field(..., description="Dane szeregu czasowego")
    model_type: str = Field(..., description="Typ modelu (linear, nonlinear, reservoir)")
    config: Optional[Dict[str, Any]] = Field(None, description="Konfiguracja modelu")

class ModelPredictRequest(BaseModel):
    """Model żądania predykcji modelu."""
    parameters: Dict[str, Any] = Field(..., description="Parametry modelu")
    model_type: str = Field(..., description="Typ modelu (linear, nonlinear, reservoir)")
    initial_conditions: Dict[str, float] = Field(..., description="Warunki początkowe")
    time_points: List[float] = Field(..., description="Punkty czasowe do predykcji")
    config: Optional[Dict[str, Any]] = Field(None, description="Konfiguracja modelu")

# Funkcje pomocnicze
def create_model(model_type: str, config: Optional[Dict[str, Any]] = None):
    """Tworzy model dynamiki relacji."""
    if model_type == "linear":
        return LinearDifferentialModel(config)
    elif model_type == "nonlinear":
        return NonlinearDifferentialModel(config)
    elif model_type == "reservoir":
        return ReservoirModel(config)
    else:
        raise ValueError(f"Nieznany typ modelu: {model_type}")

# Trasy API
@router.post("/fit", status_code=status.HTTP_200_OK)
async def fit_model(request: ModelFitRequest):
    """Dopasowuje model do danych szeregu czasowego."""
    try:
        # Tworzymy odpowiedni model
        model = create_model(request.model_type, request.config)

        # Konwertujemy dane z modelu Pydantic do numpy
        data = {
            "t": np.array(request.data.t),
            "H": np.array(request.data.H),
            "A": np.array(request.data.A)
        }

        # Dopasowujemy model
        result = model.fit(data)

        # Dodajemy analizę stabilności i wyjaśnienia parametrów
        if result.get("success", False):
            try:
                stability = model.analyze_stability()
                result["stability_analysis"] = stability
            except Exception as e:
                logger.warning(f"Błąd podczas analizy stabilności: {e}")
                result["stability_analysis"] = {"error": str(e)}

            try:
                explanations = model.get_parameter_explanations()
                result["parameter_explanations"] = explanations
            except Exception as e:
                logger.warning(f"Błąd podczas generowania wyjaśnień parametrów: {e}")
                result["parameter_explanations"] = {"error": str(e)}

            try:
                equilibrium_points = model.get_equilibrium_points()
                # Konwertujemy punkty równowagi na format serializowalny
                result["equilibrium_points"] = [
                    {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                     for k, v in point.items()} 
                    for point in equilibrium_points
                ]
            except Exception as e:
                logger.warning(f"Błąd podczas obliczania punktów równowagi: {e}")
                result["equilibrium_points"] = [{"error": str(e)}]

        return result
    except Exception as e:
        logger.error(f"Błąd podczas dopasowywania modelu: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas dopasowywania modelu: {str(e)}"
        )

@router.post("/predict", status_code=status.HTTP_200_OK)
async def predict_model(request: ModelPredictRequest):
    """Przewiduje trajektorię modelu."""
    try:
        # Tworzymy odpowiedni model
        model = create_model(request.model_type, request.config)

        # Ustawiamy parametry modelu
        model.set_parameters(request.parameters)

        # Konwertujemy punkty czasowe do numpy
        t = np.array(request.time_points)

        # Przeprowadzamy predykcję
        result = model.predict(t, request.initial_conditions)

        # Konwertujemy wyniki z numpy do listy dla serializacji JSON
        serialized_result = {
            "t": request.time_points,
            "H": result["H"].tolist(),
            "A": result["A"].tolist()
        }

        return serialized_result
    except Exception as e:
        logger.error(f"Błąd podczas predykcji modelu: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas predykcji modelu: {str(e)}"
        )

@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_model(request: ModelPredictRequest):
    """Analizuje stabilność modelu."""
    try:
        # Tworzymy odpowiedni model
        model = create_model(request.model_type, request.config)

        # Ustawiamy parametry modelu
        model.set_parameters(request.parameters)

        # Przeprowadzamy analizę stabilności
        stability = model.analyze_stability()

        # Generujemy wyjaśnienia parametrów
        explanations = model.get_parameter_explanations()

        # Znajdujemy punkty równowagi
        equilibrium_points = model.get_equilibrium_points()

        # Konwertujemy punkty równowagi na format serializowalny
        serialized_equilibrium = [
            {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
             for k, v in point.items()} 
            for point in equilibrium_points
        ]

        return {
            "stability_analysis": stability,
            "parameter_explanations": explanations,
            "equilibrium_points": serialized_equilibrium
        }
    except Exception as e:
        logger.error(f"Błąd podczas analizy modelu: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas analizy modelu: {str(e)}"
        )
