import requests

from core.config import settings


def get_embedding(text: str):
    headers = {
        "Authorization": f"Bearer {settings.mistral_api_key}",
    }
    body = {
        "model": "mistral-embed",
        "input": [text],
    }

    response = requests.post(settings.mistral_embedding_url, json=body, headers=headers)

    return response.json()["data"][0]["embedding"]
