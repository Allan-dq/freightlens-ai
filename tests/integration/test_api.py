from fastapi.testclient import TestClient

from freightlens.api.app import app
from freightlens.api.dependencies import get_config
from freightlens.document_ai.ocr import OCRResult
from freightlens.schemas.document import ShipmentDocument
from freightlens.vision.video_generator import generate_video


def client_for(config):
    app.dependency_overrides[get_config] = lambda: config
    return TestClient(app)


def test_health_reconcile_and_malformed_payload(isolated_config):
    with client_for(isolated_config) as client:
        assert client.get("/health").json()["status"] == "ok"
        payload = {
            "expected": {
                "shipment_id": "S1",
                "expected_pallet_count": 8,
                "document_confidence": 0.9,
            },
            "observed": {
                "shipment_id": "S1",
                "observed_pallet_count": 7,
                "mean_detection_confidence": 0.9,
            },
        }
        response = client.post("/reconcile", json=payload)
        assert response.status_code == 200
        assert response.json()["anomaly_type"] == "MISSING_PALLET"
        payload["observed"]["shipment_id"] = "S2"
        assert client.post("/reconcile", json=payload).status_code == 422
        payload["expected"]["expected_pallet_count"] = -1
        assert client.post("/reconcile", json=payload).status_code == 422
    app.dependency_overrides.clear()


def test_upload_limits_and_unsupported_types(isolated_config):
    isolated_config.api.max_upload_mb = 1
    with client_for(isolated_config) as client:
        assert (
            client.post("/documents/analyze", files={"file": ("bad.exe", b"x")}).status_code == 415
        )
        assert client.post("/documents/analyze", files={"file": ("a.png", b"")}).status_code == 422
        assert (
            client.post(
                "/documents/analyze", files={"file": ("a.png", b"x" * (1024 * 1024 + 1))}
            ).status_code
            == 413
        )
        assert (
            client.post(
                "/videos/analyze?shipment_id=S1", files={"file": ("a.mp4", b"not a video")}
            ).status_code
            == 422
        )
    app.dependency_overrides.clear()


def test_document_route_uses_shared_analyzer(isolated_config, monkeypatch):
    def analyzer(path, config):
        assert path.read_bytes() == b"image-test-double"
        return (
            ShipmentDocument(document_id="upload", document_type="invoice", pallet_count=5),
            OCRResult(tokens=[], backend="test", is_fixture=True),
        )

    monkeypatch.setattr("freightlens.api.routes.analyze_document", analyzer)
    with client_for(isolated_config) as client:
        response = client.post(
            "/documents/analyze", files={"file": ("a.png", b"image-test-double")}
        )
        assert response.status_code == 200
        assert response.json()["document"]["pallet_count"] == 5
    app.dependency_overrides.clear()


def test_video_upload_actual_decoding(isolated_config):
    path = generate_video(isolated_config, 0)
    with client_for(isolated_config) as client:
        response = client.post(
            "/videos/analyze?shipment_id=S1",
            files={"file": ("test.mp4", path.read_bytes(), "video/mp4")},
        )
        assert response.status_code == 200
        assert response.json()["observed"]["observed_pallet_count"] == 2
        assert response.json()["fixture"]
    app.dependency_overrides.clear()
