import logging
import requests

from fastapi import APIRouter, HTTPException
from services.ollama_client import OllamaClient

router = APIRouter(prefix="/api/llm")


@router.get("/ollama/tags")
def fetch_models():
    try:
        client = OllamaClient()
        models_name = client.get_models_name()
        return {"models": models_name}

    except requests.RequestException as e:
        logging.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=f"Unable to fetch models from Ollama: {e}")


