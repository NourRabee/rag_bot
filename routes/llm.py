import logging

import httpx

from fastapi import APIRouter, HTTPException
from core.config import settings
from schemas.llm import LLMRequest
from services.ollama_client import OllamaClient

router = APIRouter(prefix="/api/llm")


@router.get("/ollama/tags")
def fetch_models():
    try:
        client = OllamaClient()
        models_name = client.fetch_models_name()

        return {"models": models_name}

    except httpx.RequestError as e:
        logging.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=f"Unable to fetch models from Ollama: {e}")


@router.post("/ollama/chat/completions")
def get_response(body: LLMRequest):
    try:
        client = OllamaClient()
        response = client.get_response(body.prompt, f"{settings.ollama_base_url}/api/chat",
                                       body.model, None, False)

        return {"response": response}

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
