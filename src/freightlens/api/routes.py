import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from freightlens import __version__
from freightlens.api.dependencies import get_config
from freightlens.config import Config
from freightlens.pipeline import analyze_document
from freightlens.reconciliation.rules import reconcile
from freightlens.reconciliation.state import observed_from_tracking
from freightlens.schemas.anomaly import ReconciliationResult
from freightlens.schemas.document import ShipmentDocument
from freightlens.schemas.shipment import ExpectedShipmentState, ObservedShipmentState
from freightlens.vision.tracker import track_video

router = APIRouter()
Settings = Annotated[Config, Depends(get_config)]


class ReconcileRequest(BaseModel):
    expected: ExpectedShipmentState
    observed: ObservedShipmentState


class DocumentResponse(BaseModel):
    document: ShipmentDocument
    backend: str
    fixture: bool


class VideoResponse(BaseModel):
    observed: ObservedShipmentState
    backend: str
    fixture: bool
    events: int = Field(ge=0)


def store_upload(upload: UploadFile, directory: Path, allowed: set[str], maximum: int) -> Path:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in allowed:
        raise HTTPException(415, f"Allowed extensions: {', '.join(sorted(allowed))}")
    target = directory / f"upload{extension}"
    total = 0
    with target.open("wb") as output:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > maximum:
                raise HTTPException(413, "Upload exceeds configured file size limit")
            output.write(chunk)
    if total == 0:
        raise HTTPException(422, "Uploaded file is empty")
    return target


@router.get("/health")
def health():
    return {"status": "ok", "service": "freightlens", "version": __version__}


@router.post("/documents/analyze", response_model=DocumentResponse)
def document_analyze(file: UploadFile, config: Settings):
    with tempfile.TemporaryDirectory(prefix="freightlens-document-") as directory:
        path = store_upload(
            file,
            Path(directory),
            {".pdf", ".png", ".jpg", ".jpeg", ".tif"},
            config.api.max_upload_mb * 1024 * 1024,
        )
        try:
            document, ocr = analyze_document(path, config)
            return DocumentResponse(document=document, backend=ocr.backend, fixture=ocr.is_fixture)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        except (ValueError, OSError, Image.DecompressionBombError) as error:
            raise HTTPException(422, str(error)) from error


@router.post("/videos/analyze", response_model=VideoResponse)
def video_analyze(file: UploadFile, config: Settings, shipment_id: str):
    with tempfile.TemporaryDirectory(prefix="freightlens-video-") as directory:
        root = Path(directory)
        path = store_upload(
            file, root, {".mp4", ".avi", ".mov"}, config.api.max_upload_mb * 1024 * 1024
        )
        local_config = config.model_copy(deep=True)
        local_config.paths.artifacts = root / "artifacts"
        try:
            result = track_video(path, local_config, output=root / "annotated.mp4")
            return VideoResponse(
                observed=observed_from_tracking(result, shipment_id),
                backend=result["backend"],
                fixture=result["fixture"],
                events=len(result["events"]),
            )
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        except (ValueError, OSError) as error:
            raise HTTPException(422, str(error)) from error


@router.post("/reconcile", response_model=ReconciliationResult)
def reconcile_states(request: ReconcileRequest, config: Settings):
    try:
        return reconcile(request.expected, request.observed, config.reconciliation)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
