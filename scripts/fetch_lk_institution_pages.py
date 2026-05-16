#!/usr/bin/env python3
"""
Fetch public HTML from key Sri Lankan institution sites and save plain text extracts
into data/_scraped/ for optional RAG ingestion.

Usage (from ai-service/):
  pip install requests beautifulsoup4
  python scripts/fetch_lk_institution_pages.py

Respect each site's terms of use and robots.txt; run occasionally—not in hot loops.

Suggested schedule: weekly via cron, then run rag/ingest.py to rebuild FAISS.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Install dependencies: pip install requests beautifulsoup4", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("fetch_lk_institution_pages")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "_scraped"

URLS = [
    "https://www.srilankabusiness.com/",
    "https://cda.gov.lk/web/index.php?lang=en",
    "https://pdb.gov.lk/",
    "https://kdb.gov.lk/en",
]

HEADERS = {"User-Agent": "BuildBusinessLK-RAG-ingest/1.1 (+university-project)"}
REQUEST_TIMEOUT = (15, 45)
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_url(session: requests.Session, url: str) -> str | None:
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info("GET %s (attempt %s/%s)", url, attempt, MAX_RETRIES)
            response = session.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_err = exc
            log.warning("Fetch failed for %s: %s", url, exc)
            if attempt < MAX_RETRIES:
                sleep_for = RETRY_BACKOFF_SEC * attempt
                log.info("Retrying in %.1fs", sleep_for)
                time.sleep(sleep_for)
    log.error("Giving up on %s after %s attempts: %s", url, MAX_RETRIES, last_err)
    return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    chunks: list[str] = []

    session = requests.Session()

    for url in URLS:
        html = fetch_url(session, url)
        if html is None:
            chunks.append(f"== FETCH FAILED: {url} ==\nError: see logs\n")
            continue

        body = html_to_text(html)
        chunks.append(f"== SOURCE: {url} | fetched UTC {stamp} ==\n")
        chunks.append(f"Title/URL: {url}\n")
        chunks.append(body[:120000])
        chunks.append("\n")

    outfile = OUT_DIR / f"lk_institutions_scrape_{stamp}.txt"
    outfile.write_text("\n\n".join(chunks), encoding="utf-8")
    log.info("Wrote %s (%s bytes)", outfile, outfile.stat().st_size)


if __name__ == "__main__":
    main()
