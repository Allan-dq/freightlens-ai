from fastapi import FastAPI

from freightlens.api.routes import router

app = FastAPI(
    title="FreightLens AI",
    version="0.1.0",
    description="Local freight document and visual verification. "
    "Use /docs to explore uploads and reconciliation.",
)
app.include_router(router)
