from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str
    groq_base_url: str
    groq_api_key: str

    user_id: str = 'ee127586-e046-44fc-8358-7ff4030da693'
    session_id: str = 'a5bcb1c6-cd10-4f62-8504-6da4ef302b47'

    session_messages: List[dict] = Field(default_factory=list)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
