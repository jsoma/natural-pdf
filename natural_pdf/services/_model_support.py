from __future__ import annotations

CORE_COMPLETE_INSTALL = 'pip install "natural-pdf[all]"'
DOC_QA_DIRECT_INSTALL = "pip install torch transformers"
VLM_DIRECT_INSTALL = "pip install transformers torch"

DOC_QA_INSTALL_MESSAGE = (
    "Document-QA dependencies missing. Install with: "
    f"{CORE_COMPLETE_INSTALL} or {DOC_QA_DIRECT_INSTALL}"
)

QA_INSTALL_MESSAGE = (
    "Question answering requires torch and transformers. Install with: "
    f"{CORE_COMPLETE_INSTALL} or {DOC_QA_DIRECT_INSTALL}"
)

VLM_INSTALL_MESSAGE = (
    "VLM engine requires 'transformers' and 'torch'. "
    f"Install with: {CORE_COMPLETE_INSTALL} or {VLM_DIRECT_INSTALL}"
)

VISION_MODE_REQUIREMENTS = (
    "using='vision' requires either a client for LLM/VLM extraction "
    "or engine='vlm' for the local vision model."
)
