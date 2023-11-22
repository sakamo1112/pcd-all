from fastapi import FastAPI
from fastpcd.api.routes import router

app = FastAPI()
app.include_router(router)
