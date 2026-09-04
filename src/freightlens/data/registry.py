DATASETS = {
    "funsd": {"type": "public_real", "url": "https://github.com/crcresearch/FUNSD",
              "required": False, "purpose": "external_scanned_form_validation"},
    "palleset": {"type": "public_synthetic", "repo_id": "eusandre95/Palleset",
                 "license": "CC-BY-4.0", "required": True, "purpose": "pallet_detection"},
    "freight_documents": {"type": "generated", "required": True,
                          "purpose": "freight_document_extraction"},
    "tracking_scenarios": {"type": "generated", "required": True,
                           "purpose": "tracking_and_counting"},
}
