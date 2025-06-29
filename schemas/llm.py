from fastapi import UploadFile
from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    model: str
    prompt: str


class Output(BaseModel):
    response: str = Field(description="The answer of the user's question")
