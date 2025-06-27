from pydantic import BaseModel


class LLMRequest(BaseModel):
    model: str
    prompt: str
