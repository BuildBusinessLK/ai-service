#!/usr/bin/env python3
"""
Fetch public HTML from key Sri Lankan institution sites and save plain text extracts
into data/_scraped/ for optional RAG ingestion.

Usage (from ai-service/):
  pip install requests beautifulsoup4
  python scripts/fetch_lk_institution_pages.py

Respect each site's terms of use and robots.txt; run occasionally—not in hot loops.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "_scraped"

URLS = [
    "https://www.srilankabusiness.com/",
    "https://cda.gov.lk/web/index.php?lang=en",
    "https://pdb.gov.lk/",
    "https://kdb.gov.lk/en",
]


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    chunks: list[str] = []

    for url in URLS:
        try:
            response = requests.get(
                url, timeout=45, headers={"User-Agent": "BuildBusinessLK-RAG-ingest/1.0"}
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            chunks.append(f"== FETCH FAILED: {url} ==\nError: {exc}\n")
            continue

        title = url
        body = html_to_text(response.text)
        chunks.append(f"== SOURCE: {url} | fetched UTC {stamp} ==\n")
        chunks.append(f"Title/URL: {title}\n")
        chunks.append(body[:120000])
        chunks.append("\n")

    outfile = OUT_DIR / f"lk_institutions_scrape_{stamp}.txt"
    outfile.write_text("\n\n".join(chunks), encoding="utf-8")
    print(f"Wrote {outfile} ({outfile.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
