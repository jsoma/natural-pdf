"""Reuse one PDF object for Pennsylvania benchmark extraction calls."""

from contextlib import contextmanager

METADATA = {
    "track": "workflow_pdf_reuse",
    "candidate": "pennsylvania_pdf_cache",
    "cache_only": True,
    "hypothesis": "Reusing the same PDF object across pages separates workflow reopen cost from library extraction cost.",
}


@contextmanager
def install():
    import pandas as pd

    import natural_pdf as npdf
    from benchmark.configs.pennsylvania_election import PennsylvaniaElectionConfig
    from natural_pdf import Guides

    original_extract = PennsylvaniaElectionConfig.__dict__["extract_with_natural_pdf"]
    pdf_cache = {}

    def patched_extract_with_natural_pdf(pdf_path: str, page_num: int = 0) -> list:
        pdf = pdf_cache.get(pdf_path)
        if pdf is None:
            pdf = npdf.PDF(pdf_path)
            pdf_cache[pdf_path] = pdf

        if page_num >= PennsylvaniaElectionConfig.max_pages:
            return []

        dfs = []
        page = pdf.pages[page_num]
        try:
            location_elem = page.find("text[size=max()]")
            if not location_elem:
                return []
            location = location_elem.extract_text()
            positions = page.find_all("text[size=9]")
            for position_elem in positions:
                try:
                    position = position_elem.extract_text()
                    results = position_elem.below(until="text:contains(TOTAL)").below(
                        until="text:contains(Contest Totals)"
                    )
                    if not results:
                        continue
                    table_area = results.trim(method="any")
                    guides = Guides(table_area)
                    guides.vertical.divide(2)
                    guides.snap_to_whitespace()
                    guides.horizontal.from_lines(outer=True)
                    df = guides.extract_table().to_df(header=["candidate", "votes"])
                    df["votes"] = df["votes"].str.replace(",", "").astype(int)
                    df["position"] = position
                    df["location"] = location
                    dfs.append(df)
                except Exception:
                    pass
        except Exception:
            pass
        if dfs:
            return pd.concat(dfs).reset_index(drop=True).to_dict("records")
        return []

    PennsylvaniaElectionConfig.extract_with_natural_pdf = staticmethod(
        patched_extract_with_natural_pdf
    )
    try:
        yield
    finally:
        PennsylvaniaElectionConfig.extract_with_natural_pdf = original_extract
        for pdf in pdf_cache.values():
            try:
                pdf.close()
            except Exception:
                pass
        pdf_cache.clear()
