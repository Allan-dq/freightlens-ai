from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps


def load_pages(path: Path, max_pages: int = 10) -> list[np.ndarray]:
    if path.suffix.lower() == ".pdf":
        try:
            document = pdfium.PdfDocument(str(path))
        except pdfium.PdfiumError as error:
            raise ValueError("Unreadable PDF") from error
        try:
            if len(document) > max_pages:
                raise ValueError(f"PDF exceeds {max_pages} pages")
            pages = []
            for index in range(len(document)):
                page = document[index]
                if page.get_width() * page.get_height() * 2.25 > 20_000_000:
                    raise ValueError("PDF page exceeds render size limit")
                bitmap = page.render(scale=1.5)
                pages.append(np.asarray(bitmap.to_pil().convert("RGB")).copy())
                bitmap.close()
                page.close()
            return pages
        finally:
            document.close()
    with Image.open(path) as image:
        if image.width * image.height > 20_000_000:
            raise ValueError("Image exceeds 20 megapixels")
        return [np.asarray(ImageOps.exif_transpose(image).convert("RGB"))]
