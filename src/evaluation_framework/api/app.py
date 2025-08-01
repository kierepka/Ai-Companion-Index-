#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Główna aplikacja API dla frameworka AI Friendliness Evaluator

Ten moduł definiuje aplikację FastAPI, która udostępnia 
funkcjonalności frameworka przez REST API.
"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from typing import Dict, Any, List, Optional
import time
import uuid

from ..config import get_config
from ..utils.logging import setup_logger
from . import routes

logger = setup_logger("api")

def create_app() -> FastAPI:
    """Tworzy i konfiguruje aplikację FastAPI.

    Returns:
        Skonfigurowana instancja FastAPI
    """
    # Pobieramy konfigurację
    config = get_config()
    api_config = config.get("api", {})

    # Tworzymy aplikację
    app = FastAPI(
        title="AI Friendliness Evaluator API",
        description="API do oceny przyjazności systemów AI",
        version="0.1.0"
    )

    # Konfigurujemy CORS
    allowed_origins = api_config.get("allowed_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Konfigurujemy uwierzytelnianie API
    api_key_required = api_config.get("api_key_required", False)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def get_api_key(api_key: str = Depends(api_key_header)):
        """Weryfikuje klucz API."""
        if not api_key_required:
            return True

        # W rzeczywistej implementacji sprawdzalibyśmy klucz w bazie danych
        # Tutaj używamy uproszczonego podejścia z kluczem w konfiguracji
        valid_api_key = api_config.get("api_key", "")

        if not api_key or api_key != valid_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Nieprawidłowy klucz API"
            )

        return True

    # Middleware do logowania żądań
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Loguje informacje o żądaniach HTTP."""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Dodajemy ID żądania do kontekstu logów
        logger.info(f"[{request_id}] Żądanie {request.method} {request.url.path}")

        # Przetwarzamy żądanie
        response = await call_next(request)

        # Logujemy czas przetwarzania
        process_time = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Odpowiedź {response.status_code} w {process_time:.2f}ms")

        # Dodajemy ID żądania do nagłówków odpowiedzi
        response.headers["X-Request-ID"] = request_id

        return response

    # Handler błędów
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Globalny handler wyjątków."""
        logger.error(f"Nieobsłużony wyjątek: {exc}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Wystąpił wewnętrzny błąd serwera."}
        )

    # Rejestrujemy routery
    app.include_router(
        routes.evaluation.router,
        prefix="/api/v1/evaluation",
        tags=["evaluation"],
        dependencies=[Depends(get_api_key)]
    )

    app.include_router(
        routes.models.router,
        prefix="/api/v1/models",
        tags=["models"],
        dependencies=[Depends(get_api_key)]
    )

    app.include_router(
        routes.scenarios.router,
        prefix="/api/v1/scenarios",
        tags=["scenarios"],
        dependencies=[Depends(get_api_key)]
    )

    # Endpoint informacyjny
    @app.get("/", tags=["info"])
    async def root():
        """Zwraca podstawowe informacje o API."""
        return {
            "name": "AI Friendliness Evaluator API",
            "version": "0.1.0",
            "docs_url": "/docs"
        }

    # Status serwera
    @app.get("/health", tags=["info"])
    async def health_check():
        """Sprawdza stan serwera."""
        return {"status": "healthy"}

    return app
