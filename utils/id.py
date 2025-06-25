from datetime import datetime


def generate_vector_id(session_id):
    return f"{session_id}_{int(datetime.now().timestamp())}"
