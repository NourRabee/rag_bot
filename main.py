from fastapi import FastAPI
from routes import llm, files

app = FastAPI()

app.include_router(llm.router)

app.include_router(files.router)
