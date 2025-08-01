#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trasy API dla scenariuszy testowych
"""

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

from ...config import get_config
from ...utils.logging import setup_logger

logger = setup_logger("api.routes.scenarios")

router = APIRouter()

# Modele danych
class ScenarioInfo(BaseModel):
    """Model informacji o scenariuszu."""
    scenario_id: str = Field(..., description="Identyfikator scenariusza")
    description: str = Field(..., description="Opis scenariusza")
    context: Optional[str] = Field(None, description="Kontekst scenariusza")

class DialogueExchange(BaseModel):
    """Model wymiany w dialogu."""
    user: str = Field(..., description="Wypowiedź użytkownika")
    expected_emotions: Optional[List[str]] = Field(None, description="Oczekiwane emocje do rozpoznania")
    expected_response_types: Optional[List[str]] = Field(None, description="Oczekiwane typy odpowiedzi")

class ScenarioCreate(BaseModel):
    """Model tworzenia scenariusza."""
    scenario_id: str = Field(..., description="Identyfikator scenariusza")
    description: str = Field(..., description="Opis scenariusza")
    context: Optional[str] = Field(None, description="Kontekst scenariusza")
    dialogues: List[DialogueExchange] = Field(..., description="Lista wymian w dialogu")
    evaluation_criteria: Optional[Dict[str, float]] = Field(None, description="Kryteria oceny")
    negative_patterns: Optional[List[str]] = Field(None, description="Wzorce negatywne do unikania")

# Funkcje pomocnicze
def get_scenarios_dir() -> Path:
    """Zwraca ścieżkę do katalogu ze scenariuszami."""
    config = get_config()
    scenarios_dir = Path(config.get("evaluation", {}).get("scenarios_dir", "./scenarios"))
    os.makedirs(scenarios_dir, exist_ok=True)
    return scenarios_dir

# Trasy API
@router.get("/list", status_code=status.HTTP_200_OK)
async def list_scenarios():
    """Zwraca listę dostępnych scenariuszy."""
    try:
        scenarios_dir = get_scenarios_dir()
        scenario_files = list(scenarios_dir.glob("*.json"))

        scenarios = []
        for file_path in scenario_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    scenario_data = json.load(f)

                scenario_info = {
                    "scenario_id": scenario_data.get("scenario_id", file_path.stem),
                    "description": scenario_data.get("description", ""),
                    "file_name": file_path.name
                }
                scenarios.append(scenario_info)
            except Exception as e:
                logger.warning(f"Błąd podczas odczytu scenariusza {file_path}: {e}")

        return {"scenarios": scenarios}
    except Exception as e:
        logger.error(f"Błąd podczas listowania scenariuszy: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas listowania scenariuszy: {str(e)}"
        )

@router.get("/{scenario_id}", status_code=status.HTTP_200_OK)
async def get_scenario(scenario_id: str):
    """Zwraca scenariusz o podanym ID."""
    try:
        scenarios_dir = get_scenarios_dir()

        # Sprawdzamy, czy istnieje plik z dokładnym ID
        scenario_path = scenarios_dir / f"{scenario_id}.json"

        if not scenario_path.exists():
            # Szukamy pliku, który zawiera podane ID
            found = False
            for file_path in scenarios_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        scenario_data = json.load(f)

                    if scenario_data.get("scenario_id") == scenario_id:
                        scenario_path = file_path
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenariusz o ID '{scenario_id}' nie został znaleziony"
                )

        # Wczytujemy dane scenariusza
        with open(scenario_path, "r", encoding="utf-8") as f:
            scenario_data = json.load(f)

        return scenario_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas pobierania scenariusza: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas pobierania scenariusza: {str(e)}"
        )

@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_scenario(scenario: ScenarioCreate):
    """Tworzy nowy scenariusz."""
    try:
        scenarios_dir = get_scenarios_dir()
        scenario_path = scenarios_dir / f"{scenario.scenario_id}.json"

        # Sprawdzamy, czy scenariusz już istnieje
        if scenario_path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Scenariusz o ID '{scenario.scenario_id}' już istnieje"
            )

        # Tworzymy słownik z danymi scenariusza
        scenario_data = scenario.dict(exclude_none=True)

        # Zapisujemy scenariusz do pliku
        with open(scenario_path, "w", encoding="utf-8") as f:
            json.dump(scenario_data, f, indent=2, ensure_ascii=False)

        return {"message": f"Scenariusz '{scenario.scenario_id}' został utworzony", "path": str(scenario_path)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas tworzenia scenariusza: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas tworzenia scenariusza: {str(e)}"
        )

@router.put("/{scenario_id}", status_code=status.HTTP_200_OK)
async def update_scenario(scenario_id: str, scenario: ScenarioCreate):
    """Aktualizuje istniejący scenariusz."""
    try:
        scenarios_dir = get_scenarios_dir()
        scenario_path = scenarios_dir / f"{scenario_id}.json"

        # Sprawdzamy, czy scenariusz istnieje
        if not scenario_path.exists():
            # Szukamy pliku, który zawiera podane ID
            found = False
            for file_path in scenarios_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        scenario_data = json.load(f)

                    if scenario_data.get("scenario_id") == scenario_id:
                        scenario_path = file_path
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenariusz o ID '{scenario_id}' nie został znaleziony"
                )

        # Tworzymy słownik z danymi scenariusza
        scenario_data = scenario.dict(exclude_none=True)

        # Zapisujemy scenariusz do pliku
        with open(scenario_path, "w", encoding="utf-8") as f:
            json.dump(scenario_data, f, indent=2, ensure_ascii=False)

        return {"message": f"Scenariusz '{scenario_id}' został zaktualizowany", "path": str(scenario_path)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas aktualizacji scenariusza: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas aktualizacji scenariusza: {str(e)}"
        )

@router.delete("/{scenario_id}", status_code=status.HTTP_200_OK)
async def delete_scenario(scenario_id: str):
    """Usuwa scenariusz o podanym ID."""
    try:
        scenarios_dir = get_scenarios_dir()
        scenario_path = scenarios_dir / f"{scenario_id}.json"

        # Sprawdzamy, czy scenariusz istnieje
        if not scenario_path.exists():
            # Szukamy pliku, który zawiera podane ID
            found = False
            for file_path in scenarios_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        scenario_data = json.load(f)

                    if scenario_data.get("scenario_id") == scenario_id:
                        scenario_path = file_path
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenariusz o ID '{scenario_id}' nie został znaleziony"
                )

        # Usuwamy plik scenariusza
        os.remove(scenario_path)

        return {"message": f"Scenariusz '{scenario_id}' został usunięty"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas usuwania scenariusza: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas usuwania scenariusza: {str(e)}"
        )

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_scenario(
    file: UploadFile = File(...),
    replace: bool = Form(False)
):
    """Wgrywa scenariusz z pliku JSON."""
    try:
        scenarios_dir = get_scenarios_dir()
        file_path = scenarios_dir / file.filename

        # Sprawdzamy, czy plik już istnieje
        if file_path.exists() and not replace:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Plik '{file.filename}' już istnieje"
            )

        # Wczytujemy zawartość pliku
        content = await file.read()

        # Sprawdzamy, czy zawartość to poprawny JSON
        try:
            scenario_data = json.loads(content)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plik nie zawiera poprawnego formatu JSON"
            )

        # Sprawdzamy, czy zawiera wymagane pola
        if "scenario_id" not in scenario_data or "dialogues" not in scenario_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Scenariusz musi zawierać pola 'scenario_id' i 'dialogues'"
            )

        # Zapisujemy plik
        with open(file_path, "wb") as f:
            f.write(content)

        return {
            "message": f"Scenariusz '{file.filename}' został wgrany",
            "scenario_id": scenario_data.get("scenario_id"),
            "path": str(file_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas wgrywania scenariusza: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas wgrywania scenariusza: {str(e)}"
        )

@router.get("/download/{scenario_id}", status_code=status.HTTP_200_OK)
async def download_scenario(scenario_id: str):
    """Pobiera scenariusz jako plik JSON."""
    try:
        scenarios_dir = get_scenarios_dir()
        scenario_path = scenarios_dir / f"{scenario_id}.json"

        # Sprawdzamy, czy scenariusz istnieje
        if not scenario_path.exists():
            # Szukamy pliku, który zawiera podane ID
            found = False
            for file_path in scenarios_dir.glob("*.json"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        scenario_data = json.load(f)

                    if scenario_data.get("scenario_id") == scenario_id:
                        scenario_path = file_path
                        found = True
                        break
                except Exception:
                    continue

            if not found:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Scenariusz o ID '{scenario_id}' nie został znaleziony"
                )

        return FileResponse(
            path=scenario_path,
            filename=scenario_path.name,
            media_type="application/json"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas pobierania pliku scenariusza: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas pobierania pliku scenariusza: {str(e)}"
        )
