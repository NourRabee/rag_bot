from fastapi import FastAPI
from routes import llm

app = FastAPI()

app.include_router(llm.router)


