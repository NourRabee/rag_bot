import logging

import httpx

from fastapi import APIRouter, HTTPException
from core.config import settings
from schemas.llm import LLMRequest
from services.llm_client import LLMClient

router = APIRouter(prefix="/api/llm")

llm_client = LLMClient()


@router.get("/ollama/tags")
def fetch_models():
    try:
        models_name = llm_client.fetch_ollama_models()

        return {"models": models_name}

    except httpx.RequestError as e:
        logging.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=f"Unable to fetch models from Ollama: {e}")


@router.post("/ollama/chat/completions")
def get_response(body: LLMRequest):
    try:
        response = llm_client.handle_user_query(settings.session_id, settings.user_id, body.prompt, body.model,
                                                "ollama")

        return response

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@router.post("/groq/chat/completions")
def get_response(body: LLMRequest):
    try:
        response = llm_client.handle_user_query(settings.session_id, settings.user_id, body.prompt,
                                                body.model, "groq")

        return response

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
