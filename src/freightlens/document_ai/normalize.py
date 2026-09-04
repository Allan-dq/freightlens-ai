import re
import unicodedata


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).upper().split())


def normalize_vehicle(value: str | None) -> str | None:
    return re.sub(r"[^A-Z0-9]", "", normalize_text(value)) or None


def parse_number(value: str, integer: bool = False) -> int | float:
    match = re.fullmatch(
        r"\s*(\d[\d,]*(?:\.\d+)?)\s*(lb|lbs|kg|pallets?|packages?)?\s*", value, flags=re.IGNORECASE
    )
    if not match:
        raise ValueError(f"Invalid numeric field: {value}")
    number = float(match[1].replace(",", ""))
    if (match[2] or "").lower() == "kg":
        number *= 2.2046226218
    if integer:
        if not number.is_integer():
            raise ValueError("Count must be integral")
        return int(number)
    return round(number, 2)
