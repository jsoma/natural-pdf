"""
Per-PDF benchmark configurations.

Each PDF gets its own config file that defines:
1. How to extract with Natural PDF (ground truth)
2. LLM prompts for different extraction tasks
3. Expected fields/structure
"""

from benchmark.configs.arkansas_state import ArkansasStateConfig
from benchmark.configs.atlanta_schools import AtlantaSchoolsConfig
from benchmark.configs.guides_expenses import GuidesExpensesConfig
from benchmark.configs.hebrew_table import HebrewTableConfig
from benchmark.configs.oklahoma_license import OklahomaLicenseConfig
from benchmark.configs.pennsylvania_election import PennsylvaniaElectionConfig
from benchmark.configs.practice_01 import Practice01Config

ALL_CONFIGS = {
    "01-practice": Practice01Config,
    "Atlanta_Public_Schools_GA_sample": AtlantaSchoolsConfig,
    # "30": ArkansasStateConfig,
    "m27": OklahomaLicenseConfig,
    "guides-expenses-sample": GuidesExpensesConfig,
    "hebrew-table": HebrewTableConfig,
    "0500000US42001": PennsylvaniaElectionConfig,
    "pennsylvania-election": PennsylvaniaElectionConfig,  # Alias
}


def get_config(pdf_name: str):
    """Get config for a PDF by name."""
    clean_name = (
        pdf_name.replace("pdfs/", "")
        .replace(".pdf", "")
        .replace("-benchmark", "")
        .replace("-trap", "")
    )
    # Try direct match first
    if clean_name in ALL_CONFIGS:
        return ALL_CONFIGS.get(clean_name)
    # Try case-insensitive match
    lower_name = clean_name.lower()
    for key, config in ALL_CONFIGS.items():
        if key.lower() == lower_name:
            return config
    return None
