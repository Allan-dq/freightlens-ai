"""Small public FUNSD download with safe ZIP extraction and provenance."""

import io
import zipfile

import httpx

from freightlens.config import Config
from freightlens.io import write_json

FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"


def download_funsd(config: Config) -> dict:
    output = config.paths.data / "raw/funsd"
    buffer = io.BytesIO()
    with httpx.stream("GET", FUNSD_URL, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            buffer.write(chunk)
            if buffer.tell() > 100 * 1024 * 1024:
                raise ValueError("FUNSD archive exceeds 100 MB budget")
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(buffer) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()
                   and "__MACOSX" not in info.filename and not info.filename.split("/")[-1].startswith("._")]
        if sum(info.file_size for info in members) > 300 * 1024 * 1024:
            raise ValueError("Uncompressed FUNSD exceeds size budget")
        for info in members:
            destination = (output / info.filename).resolve()
            if not destination.is_relative_to(output.resolve()):
                raise ValueError("Unsafe path in archive")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read(info))
    result = {"source": FUNSD_URL, "files": len(members), "download_bytes": buffer.tell(),
              "purpose": "external real scanned form validation; not freight"}
    write_json(config.paths.data / "manifests/funsd.json", result)
    return result
