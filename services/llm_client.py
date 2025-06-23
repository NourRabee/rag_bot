from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    @abstractmethod
    async def get_response(
            self,
            prompt: str,
            uri: str,
            model: str,
            api_key: Optional[str] = None,
            stream: Optional[bool] = False) -> str:
        pass
