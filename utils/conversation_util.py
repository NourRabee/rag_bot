from core.config import settings


def add_to_conversation(role: str, content: str):
    settings.session_messages.append(
        {
            "role": role,
            "content": content
        }
    )
