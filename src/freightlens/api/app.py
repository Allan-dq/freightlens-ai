from fastapi import FastAPI

from freightlens import __version__
from freightlens.api.routes import router

app = FastAPI(
    title="FreightLens AI",
    version=__version__,
    description="Local freight document and visual verification. "
    "Use /docs to explore uploads and reconciliation.",
)
app.include_router(router)
