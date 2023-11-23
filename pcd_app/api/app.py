from fastapi import FastAPI
from fastpcd.api.routes import router  # type: ignore

app = FastAPI()
app.include_router(router)
