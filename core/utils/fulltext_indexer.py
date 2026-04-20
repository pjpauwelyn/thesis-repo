"""full-text indexer for ontology-driven RAG augmentation.

pipeline (all deterministic, resumable, cache-backed):
    uri -> openalex work id -> metadata (pdf_url) -> pdf -> extracted text
    -> section-aware chunks -> scored against ontology -> top-K selected

caches live under FULLTEXT_CACHE_DIR (default: cache/fulltext/) with three
sub-folders:
    openalex/<work_id>.json        metadata
    pdfs/<work_id>.pdf             raw pdf
    text/<work_id>.json            extracted sections

usage (as a library):
    from core.utils.fulltext_indexer import FullTextIndexer
    fti = FullTextIndexer(mailto="pjpauwelyn@gmail.com")
    fti.build_document_cache(uris)
    excerpts = fti.select_excerpts_for_question(
        question=q_text,
        ontology=ont,
        documents=docs,  # list of {title, abstract, uri}
        per_doc_budget=6000,
        global_budget=90000,
        top_k_per_doc=10,
    )

usage (as a CLI):
    python3 -m core.utils.fulltext_indexer prebuild --input data/dlr/...csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = Path(os.getenv("FULLTEXT_CACHE_DIR", "cache/fulltext"))
DEFAULT_MAILTO = os.getenv("FULLTEXT_MAILTO", "pjpauwelyn@gmail.com")

# browser-like UA: many publishers (Wiley, OUP, IOP, Nature) 403 non-browser
# requests. polite pool for OpenAlex/Unpaywall uses mailto param instead.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
API_UA = f"thesis-rag-fulltext/1.0 (mailto:{DEFAULT_MAILTO})"

OPENALEX_WORKS = "https://api.openalex.org/works/"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2/"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1/paper/"

# publisher-specific hints: fetch the landing page with these to make them
# serve the PDF redirect more reliably
_PDF_ACCEPT = "application/pdf,text/html;q=0.9,*/*;q=0.1"

_CITATION_PDF_META_RE = re.compile(
    r'<meta\s+[^>]*name="citation_pdf_url"[^>]*content="([^"]+)"',
    re.IGNORECASE,
)
_CITATION_PDF_FULLTEXT_RE = re.compile(
    r'<meta\s+[^>]*name="citation_pdf_fulltext_url"[^>]*content="([^"]+)"',
    re.IGNORECASE,
)
_CITATION_PDF_META_ALT_RE = re.compile(
    r'<meta\s+[^>]*content="([^"]+)"[^>]*name="citation_pdf_url"',
    re.IGNORECASE,
)

# section priorities (multiplicative boost on chunk score)
_SECTION_BOOSTS: Dict[str, float] = {
    "methods": 1.5, "method": 1.5, "materials and methods": 1.5,
    "data and methods": 1.5, "study area": 1.2, "experiment": 1.4,
    "results": 1.5, "result": 1.5,
    "discussion": 1.3, "analysis": 1.3,
    "conclusion": 1.1, "conclusions": 1.1,
    "abstract": 1.0, "introduction": 0.6, "background": 0.7,
    "references": 0.2, "bibliography": 0.2, "acknowledg": 0.2,
    "appendix": 0.9,
    "unknown": 1.0,
}

# units / numeric boosts
_UNIT_TOKENS = {
    "mm", "cm", "m", "km", "ha", "m2", "km2", "kg", "g", "t",
    "°c", "c", "k", "%", "mm/yr", "cm/yr", "m/s", "w/m2", "gpp",
    "gt", "tg", "mg", "µm", "um", "nm", "ppm", "ppb", "yr", "yr-1",
    "day", "days", "year", "years", "ndvi", "lai", "et",
}

_NUMERIC_RE = re.compile(r"(?<![A-Za-z_])-?\d+(?:[.,]\d+)?(?:e[+-]?\d+)?(?![A-Za-z_])", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}")


# ---------------------------------------------------------------------------
# data classes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Playwright fetcher (lazy, single Chromium instance for the whole run)
# ---------------------------------------------------------------------------

class _PlaywrightFetcher:
    """headless Chromium wrapper. only used as last-resort fallback for
    Cloudflare-gated publishers. keeps ONE browser open for the whole run;
    each fetch opens a fresh context so cookies/cache don't accumulate."""

    def __init__(self, timeout_ms: int = 45000, max_bytes: int = 40 * 1024 * 1024):
        self.timeout_ms = timeout_ms
        self.max_bytes = max_bytes
        self._pw = None
        self._browser = None
        self._started = False

    def _start(self):
        if self._started:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "playwright not installed. run: pip install playwright && python -m playwright install chromium"
            ) from e
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        self._started = True

    def fetch(self, url: str) -> Optional[bytes]:
        self._start()
        ctx = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            accept_downloads=True,
            viewport={"width": 1280, "height": 900},
        )
        # hide automation hints
        ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
        )
        page = ctx.new_page()
        try:
            # strategy: first try direct request via page.request (keeps cookies
            # from the CF challenge if any). navigate first to seed cookies.
            host = url.split("/")[2] if "//" in url else ""
            if host:
                try:
                    page.goto(f"https://{host}/", timeout=self.timeout_ms, wait_until="domcontentloaded")
                    page.wait_for_timeout(500)
                except Exception:
                    pass
            # now fetch the PDF URL directly
            try:
                resp = ctx.request.get(url, timeout=self.timeout_ms)
                if resp.ok:
                    body = resp.body()
                    # verify it's actually a PDF
                    if body[:4] == b"%PDF" and len(body) <= self.max_bytes:
                        return body
                    # if HTML, try to extract citation_pdf_url and retry once
                    if b"<html" in body[:4000].lower() or b"<!doctype" in body[:4000].lower():
                        html = body.decode("utf-8", errors="ignore")
                        m = re.search(
                            r'<meta[^>]+name="citation_pdf_url"[^>]+content="([^"]+)"',
                            html, re.IGNORECASE,
                        ) or re.search(
                            r'<meta[^>]+content="([^"]+)"[^>]+name="citation_pdf_url"',
                            html, re.IGNORECASE,
                        )
                        if m:
                            alt = m.group(1).strip()
                            if alt.startswith("//"):
                                alt = "https:" + alt
                            resp2 = ctx.request.get(alt, timeout=self.timeout_ms)
                            if resp2.ok:
                                body2 = resp2.body()
                                if body2[:4] == b"%PDF" and len(body2) <= self.max_bytes:
                                    return body2
            except Exception:
                pass
            # fallback: render the page and capture a PDF download if triggered
            return None
        finally:
            try:
                page.close()
            except Exception:
                pass
            try:
                ctx.close()
            except Exception:
                pass

    def close(self):
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        self._started = False


@dataclass
class Chunk:
    work_id: str
    section: str
    page: int
    text: str
    tokens: int
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_id": self.work_id,
            "section": self.section,
            "page": self.page,
            "text": self.text,
            "tokens": self.tokens,
            "score": round(self.score, 3),
        }


@dataclass
class ExtractedDoc:
    work_id: str
    sections: List[Dict[str, Any]] = field(default_factory=list)  # [{heading, text, page}]
    pages: int = 0
    source_pdf_url: str = ""
    extraction_ok: bool = False
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_id": self.work_id,
            "sections": self.sections,
            "pages": self.pages,
            "source_pdf_url": self.source_pdf_url,
            "extraction_ok": self.extraction_ok,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _work_id_from_uri(uri: str) -> Optional[str]:
    """extract openalex work id from uri (e.g. 'W2061234567')."""
    if not uri or "openalex.org" not in uri:
        return None
    wid = uri.rstrip("/").split("/")[-1]
    if not wid or not wid.startswith("W"):
        return None
    return wid


def _estimate_tokens(text: str) -> int:
    """rough char/4 heuristic; deterministic."""
    return max(1, len(text) // 4)


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# indexer
# ---------------------------------------------------------------------------

class FullTextIndexer:
    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        mailto: str = DEFAULT_MAILTO,
        request_timeout: int = 30,
        max_pdf_mb: int = 40,
        use_browser: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.mailto = mailto
        self.request_timeout = request_timeout
        self.max_pdf_bytes = max_pdf_mb * 1024 * 1024
        self.use_browser = use_browser
        self._browser_fetcher = None  # lazy init
        # serialize Playwright calls across threads (Chromium context is not
        # safe to share from multiple python threads; HTTP fetches remain fully
        # parallel because requests.Session is thread-safe for disjoint hosts)
        self._browser_lock = threading.Lock()

        self.meta_dir = self.cache_dir / "openalex"
        self.pdf_dir = self.cache_dir / "pdfs"
        self.text_dir = self.cache_dir / "text"
        for d in (self.meta_dir, self.pdf_dir, self.text_dir):
            d.mkdir(parents=True, exist_ok=True)

        # session used for API calls (OpenAlex, Unpaywall) — polite UA
        self.api_session = requests.Session()
        self.api_session.headers.update({"User-Agent": API_UA})
        # session used for PDF/HTML fetches — browser UA to avoid 403s
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": BROWSER_UA,
            "Accept": _PDF_ACCEPT,
            "Accept-Language": "en-US,en;q=0.9",
        })

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    def fetch_metadata(self, work_id: str) -> Optional[Dict[str, Any]]:
        """fetch openalex metadata, cache to disk."""
        cache_path = self.meta_dir / f"{work_id}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        url = f"{OPENALEX_WORKS}{work_id}?mailto={self.mailto}"
        for attempt in range(3):
            try:
                resp = self.api_session.get(url, timeout=self.request_timeout)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
                # keep only the fields we need (still small)
                trimmed = {
                    "id": data.get("id"),
                    "doi": data.get("doi"),
                    "title": data.get("title"),
                    "publication_year": data.get("publication_year"),
                    "open_access": data.get("open_access") or {},
                    "best_oa_location": data.get("best_oa_location") or {},
                    "primary_location": data.get("primary_location") or {},
                    "locations": data.get("locations") or [],
                    "has_fulltext": data.get("has_fulltext", False),
                }
                cache_path.write_text(json.dumps(trimmed), encoding="utf-8")
                return trimmed
            except requests.RequestException as exc:
                if attempt == 2:
                    logger.warning(f"openalex fetch failed {work_id}: {exc}")
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    @staticmethod
    def _candidate_pdf_urls(meta: Dict[str, Any]) -> List[str]:
        """return an ordered list of candidate URLs (PDFs first, then landing
        pages). tries every location OpenAlex exposes, not just the first."""
        candidates: List[str] = []
        # best_oa / primary first
        for key in ("best_oa_location", "primary_location"):
            loc = meta.get(key) or {}
            if loc.get("pdf_url"):
                candidates.append(loc["pdf_url"])
            if loc.get("landing_page_url"):
                candidates.append(loc["landing_page_url"])
        # then every other location (preprints, institutional repos)
        for loc in (meta.get("locations") or []):
            if not isinstance(loc, dict):
                continue
            if loc.get("pdf_url"):
                candidates.append(loc["pdf_url"])
            if loc.get("landing_page_url"):
                candidates.append(loc["landing_page_url"])
        # the top-level oa_url (often matches best_oa)
        oa = meta.get("open_access") or {}
        if oa.get("oa_url"):
            candidates.append(oa["oa_url"])

        # de-dup keeping order, PDFs prioritised
        seen = set()
        pdfs = [c for c in candidates if c.lower().endswith(".pdf") and not (c in seen or seen.add(c))]
        pages = [c for c in candidates if not c.lower().endswith(".pdf") and not (c in seen or seen.add(c))]
        return pdfs + pages

    def _semantic_scholar_pdf_url(
        self, doi: Optional[str], work_id: Optional[str]
    ) -> Optional[str]:
        """query Semantic Scholar for an openAccessPdf URL.
        supports lookup by DOI or OpenAlex ID."""
        identifier = None
        if doi:
            clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
            if clean:
                identifier = f"DOI:{clean}"
        if not identifier and work_id:
            identifier = f"OPENALEX:{work_id}"
        if not identifier:
            return None
        url = f"{SEMANTIC_SCHOLAR_BASE}{identifier}?fields=openAccessPdf"
        try:
            resp = self.api_session.get(url, timeout=self.request_timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None
        pdf = (data.get("openAccessPdf") or {}).get("url")
        return pdf

    def _unpaywall_pdf_url(self, doi: Optional[str]) -> Optional[str]:
        if not doi:
            return None
        # normalise to bare DOI
        doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        if not doi_clean:
            return None
        url = f"{UNPAYWALL_BASE}{doi_clean}?email={self.mailto}"
        try:
            resp = self.api_session.get(url, timeout=self.request_timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None
        # prefer best_oa_location, then oa_locations
        best = data.get("best_oa_location") or {}
        if best.get("url_for_pdf"):
            return best["url_for_pdf"]
        for loc in (data.get("oa_locations") or []):
            if isinstance(loc, dict) and loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
        return None

    def _resolve_pdf_from_html(self, html_text: str, base_url: str) -> Optional[str]:
        """parse <meta name='citation_pdf_url'> (or alt attribute orders)
        from HTML; fall back to first <a href='*.pdf'> if nothing else."""
        if not html_text:
            return None
        for rx in (_CITATION_PDF_META_RE, _CITATION_PDF_FULLTEXT_RE, _CITATION_PDF_META_ALT_RE):
            m = rx.search(html_text)
            if m:
                url = self._abs_url(m.group(1).strip(), base_url)
                if url:
                    return url
        # weak fallback: any anchor ending in .pdf
        m = re.search(r'href="([^"]+\.pdf[^"]*)"', html_text, re.IGNORECASE)
        if m:
            return self._abs_url(m.group(1).strip(), base_url)
        return None

    @staticmethod
    def _abs_url(url: str, base_url: str) -> Optional[str]:
        if not url:
            return None
        if url.startswith("//"):
            return "https:" + url
        if url.startswith("/") and "://" in base_url:
            root = "/".join(base_url.split("/")[:3])
            return root + url
        if url.startswith("http"):
            return url
        return None

    @staticmethod
    def _host_rewrites(url: str) -> List[str]:
        """publisher-specific URL rewrites that often expose the PDF directly.
        returns a list of ALTERNATIVE URLs to try (before the original)."""
        out: List[str] = []
        low = url.lower()
        # MDPI: https://www.mdpi.com/<issn>/<vol>/<issue>/<art> -> +/pdf
        if "mdpi.com/" in low and not low.rstrip("/").endswith("/pdf"):
            clean = url.split("#")[0].split("?")[0].rstrip("/")
            if not clean.endswith(".pdf"):
                out.append(clean + "/pdf")
        # doiserbia landing pages often have a PDF link at /aid-* pattern
        # copernicus: landing like <journal>.copernicus.org/articles/<vol>/<p>
        if ".copernicus.org/articles/" in low and not low.endswith(".pdf"):
            m = re.match(r"(https?://[^/]+/articles/\d+/\d+/\d+/)", url)
            if m:
                root = m.group(1)
                # e.g. esurf-1-53-2013.pdf  (need journal slug)
                # extract journal from host
                host = url.split("/")[2]
                slug = host.split(".")[0]  # esurf, tc, bg
                # pull year+volume+page from URL
                m2 = re.match(r".+/articles/(\d+)/(\d+)/(\d+)/", url)
                if m2:
                    vol, page, year = m2.group(1), m2.group(2), m2.group(3)
                    out.append(f"{root}{slug}-{vol}-{page}-{year}.pdf")
        return out

    # ------------------------------------------------------------------
    # pdf download
    # ------------------------------------------------------------------

    @staticmethod
    def _per_request_headers(url: str) -> Dict[str, str]:
        """fuller set of browser-like headers, including a sensible Referer.
        helps get past CDN bot-shields (Cloudflare/Imperva) on MDPI etc."""
        host = url.split("/")[2] if "//" in url else ""
        referer = f"https://{host}/" if host else "https://www.google.com/"
        return {
            "Referer": referer,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

    def _try_fetch_pdf(self, url: str, cache_path: Path, _depth: int = 0) -> Tuple[bool, str]:
        """attempt to fetch ONE candidate URL. returns (success, reason).

        if the response is HTML, parses <meta citation_pdf_url> and recurses
        (up to _depth 2) to follow publisher landing pages reliably."""
        if _depth > 2:
            return False, "too_many_redirects"
        try:
            hdrs = self._per_request_headers(url)
            with self.session.get(
                url, timeout=self.request_timeout, stream=True,
                allow_redirects=True, headers=hdrs,
            ) as resp:
                if resp.status_code >= 400:
                    return False, f"http_{resp.status_code}"
                ctype = (resp.headers.get("content-type") or "").lower()
                final_url = resp.url or url
                # HTML: try to resolve to a real PDF URL
                if "pdf" not in ctype and not final_url.lower().endswith(".pdf"):
                    if "text/html" in ctype or "xml" in ctype:
                        # re-fetch as a plain (non-stream) request so requests
                        # handles gzip/deflate decompression and encoding.
                        try:
                            html_resp = self.session.get(
                                final_url, timeout=self.request_timeout,
                                allow_redirects=True,
                                headers=self._per_request_headers(final_url),
                            )
                            html_text = html_resp.text[: 1024 * 1024]
                        except Exception:
                            html_text = ""
                        resolved = self._resolve_pdf_from_html(html_text, final_url)
                        if resolved and resolved != url and resolved != final_url:
                            return self._try_fetch_pdf(resolved, cache_path, _depth + 1)
                    return False, f"not_pdf:{ctype[:30]}"
                # stream the PDF
                total = 0
                tmp = cache_path.with_suffix(".pdf.part")
                with open(tmp, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > self.max_pdf_bytes:
                            out.close()
                            tmp.unlink(missing_ok=True)
                            return False, "too_large"
                        out.write(chunk)
                tmp.rename(cache_path)
                return True, "ok"
        except Exception as exc:
            return False, f"exc:{type(exc).__name__}"

    def download_pdf(
        self, work_id: str, candidates: List[str], doi: Optional[str] = None
    ) -> Tuple[Optional[Path], str]:
        """try every candidate URL; on full failure, ask Unpaywall.
        returns (pdf_path or None, status string)."""
        cache_path = self.pdf_dir / f"{work_id}.pdf"
        fail_marker = self.pdf_dir / f"{work_id}.fail"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path, "ok"
        if fail_marker.exists():
            return None, "already_failed"

        reasons: List[str] = []
        # expand each candidate with host-specific rewrites first
        expanded: List[str] = []
        for url in candidates:
            if not url:
                continue
            for alt in self._host_rewrites(url):
                if alt not in expanded:
                    expanded.append(alt)
            if url not in expanded:
                expanded.append(url)

        for url in expanded:
            ok, why = self._try_fetch_pdf(url, cache_path)
            if ok:
                return cache_path, "ok"
            reasons.append(f"{url[:60]}->{why}")

        # Unpaywall fallback
        unpaywall_url = self._unpaywall_pdf_url(doi)
        if unpaywall_url and unpaywall_url not in expanded:
            ok, why = self._try_fetch_pdf(unpaywall_url, cache_path)
            if ok:
                return cache_path, "ok_unpaywall"
            reasons.append(f"unpaywall:{why}")

        # Semantic Scholar fallback (aggregates multiple OA repos)
        s2_url = self._semantic_scholar_pdf_url(doi, work_id)
        if s2_url and s2_url not in expanded and s2_url != unpaywall_url:
            ok, why = self._try_fetch_pdf(s2_url, cache_path)
            if ok:
                return cache_path, "ok_s2"
            reasons.append(f"s2:{why}")

        # Playwright fallback (Cloudflare-gated hosts: Wiley, MDPI, OUP, IOP,
        # Tandfonline, AnnualReviews, RoyalSociety). Expensive but works.
        if self.use_browser:
            # try best pdf url + landing page via real browser
            browser_urls: List[str] = []
            for u in expanded[:4]:  # cap attempts
                if u and u not in browser_urls:
                    browser_urls.append(u)
            if s2_url and s2_url not in browser_urls:
                browser_urls.append(s2_url)
            if unpaywall_url and unpaywall_url not in browser_urls:
                browser_urls.append(unpaywall_url)
            for burl in browser_urls:
                data = self._browser_fetch(burl)
                if data:
                    cache_path.write_bytes(data)
                    return cache_path, "ok_browser"
            reasons.append("browser_all_failed")

        fail_marker.write_text(" | ".join(reasons)[:400] or "no_candidates")
        return None, "pdf_download_failed"

    # ------------------------------------------------------------------
    # Playwright fallback (lazy; only instantiated when use_browser=True)
    # ------------------------------------------------------------------

    def _browser_fetch(self, url: str) -> Optional[bytes]:
        """fetch a PDF via a real headless Chromium. returns raw bytes or None.
        Serialized with a lock because Playwright's sync API is not safe to
        call concurrently from multiple python threads."""
        try:
            with self._browser_lock:
                if self._browser_fetcher is None:
                    self._browser_fetcher = _PlaywrightFetcher(
                        timeout_ms=max(self.request_timeout * 1500, 45000),
                        max_bytes=self.max_pdf_bytes,
                    )
                return self._browser_fetcher.fetch(url)
        except Exception as exc:
            logger.warning(f"browser fetch failed for {url}: {exc}")
            return None

    def close(self):
        if self._browser_fetcher is not None:
            try:
                self._browser_fetcher.close()
            except Exception:
                pass
            self._browser_fetcher = None

    # ------------------------------------------------------------------
    # text extraction (pymupdf)
    # ------------------------------------------------------------------

    def extract_text(self, work_id: str, pdf_path: Path) -> ExtractedDoc:
        """extract text per page + light heading detection.

        heading heuristic: a short line (≤80 chars) in title-case or all-caps
        that looks like a known section keyword, or is font-larger (when pymupdf
        exposes block font sizes); otherwise inherit the previous heading.
        """
        cache_path = self.text_dir / f"{work_id}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                doc = ExtractedDoc(work_id=work_id)
                doc.sections = data.get("sections", [])
                doc.pages = data.get("pages", 0)
                doc.extraction_ok = data.get("extraction_ok", False)
                doc.source_pdf_url = data.get("source_pdf_url", "")
                doc.error = data.get("error", "")
                return doc
            except Exception:
                pass

        doc = ExtractedDoc(work_id=work_id)
        try:
            import fitz  # pymupdf
        except ImportError:
            doc.error = "pymupdf_missing"
            return doc

        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as exc:
            doc.error = f"open_failed:{exc}"
            cache_path.write_text(json.dumps(doc.to_dict()), encoding="utf-8")
            return doc

        current_heading = "unknown"
        sections: List[Dict[str, Any]] = []
        # buffer lines per heading; flush when heading changes
        buf: List[str] = []
        buf_page_start = 1

        def flush(page_end: int):
            if buf:
                sections.append({
                    "heading": current_heading,
                    "text": " ".join(buf).strip(),
                    "page_start": buf_page_start,
                    "page_end": page_end,
                })

        try:
            for pi, page in enumerate(pdf, start=1):
                try:
                    blocks = page.get_text("blocks") or []
                except Exception:
                    blocks = []
                for b in sorted(blocks, key=lambda x: (x[1], x[0])):
                    if len(b) < 5:
                        continue
                    text = (b[4] or "").strip()
                    if not text:
                        continue
                    # every physical line
                    for raw_line in text.split("\n"):
                        line = raw_line.strip()
                        if not line:
                            continue
                        maybe_heading = self._detect_heading(line)
                        if maybe_heading is not None:
                            flush(pi)
                            current_heading = maybe_heading
                            buf = []
                            buf_page_start = pi
                        else:
                            buf.append(line)
            flush(pdf.page_count)
            doc.pages = pdf.page_count
            doc.sections = sections
            doc.extraction_ok = any(s.get("text") for s in sections)
        except Exception as exc:
            doc.error = f"extract_failed:{exc}"
        finally:
            try:
                pdf.close()
            except Exception:
                pass

        cache_path.write_text(json.dumps(doc.to_dict()), encoding="utf-8")
        return doc

    @staticmethod
    def _detect_heading(line: str) -> Optional[str]:
        """return a normalised section name if line looks like a heading."""
        if not line or len(line) > 90:
            return None
        stripped = line.strip(" .:")
        # numbered sections like "2. Methods" or "3.1 Study area"
        m = re.match(r"^(?:[0-9IVX]+\.?[0-9\.]*)\s+([A-Z][A-Za-z \-/]{2,60})$", stripped)
        candidate = m.group(1) if m else stripped
        low = candidate.lower().strip()
        # exact / prefix matches against known keywords
        for key in _SECTION_BOOSTS.keys():
            if key == "unknown":
                continue
            if low == key or low.startswith(key):
                return key
        # all-caps short heading
        if stripped.isupper() and 3 <= len(stripped) <= 60 and not stripped.endswith("."):
            low2 = stripped.lower()
            for key in _SECTION_BOOSTS.keys():
                if key in low2:
                    return key
        return None

    # ------------------------------------------------------------------
    # chunking
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_sections(
        doc: ExtractedDoc,
        target_tokens: int = 600,
        overlap_tokens: int = 60,
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        for sec in doc.sections:
            heading = (sec.get("heading") or "unknown").lower()
            text = sec.get("text") or ""
            page = sec.get("page_start", 1)
            if not text:
                continue
            # split into sentences-ish
            sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
            cur: List[str] = []
            cur_tok = 0
            for s in sents:
                t = _estimate_tokens(s)
                if cur_tok + t > target_tokens and cur:
                    chunks.append(Chunk(
                        work_id=doc.work_id, section=heading, page=page,
                        text=" ".join(cur).strip(), tokens=cur_tok,
                    ))
                    # overlap: keep tail sentences up to overlap_tokens
                    tail: List[str] = []
                    tail_tok = 0
                    for sent in reversed(cur):
                        tt = _estimate_tokens(sent)
                        if tail_tok + tt > overlap_tokens:
                            break
                        tail.insert(0, sent)
                        tail_tok += tt
                    cur = list(tail)
                    cur_tok = tail_tok
                cur.append(s)
                cur_tok += t
            if cur:
                chunks.append(Chunk(
                    work_id=doc.work_id, section=heading, page=page,
                    text=" ".join(cur).strip(), tokens=cur_tok,
                ))
        return chunks

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query_terms(question: str, ontology: Any) -> Tuple[List[str], List[str]]:
        """return (keyword_terms, literal_values). literal_values are substrings
        we want to match case-insensitively (specific units / numeric literals /
        named entities from ontology)."""
        kws: List[str] = list(_tokenize(question))
        lits: List[str] = []
        if ontology is not None:
            pairs = getattr(ontology, "attribute_value_pairs", None) or []
            for av in pairs:
                for attr_name in ("attribute", "value", "description"):
                    val = getattr(av, attr_name, None)
                    if val and isinstance(val, str):
                        kws.extend(_tokenize(val))
                        # literal substrings (unit or short phrase)
                        short = val.strip()
                        if short and len(short) <= 40:
                            lits.append(short.lower())
        # dedup keeping order
        seen = set()
        kws = [k for k in kws if not (k in seen or seen.add(k)) and len(k) >= 3]
        seen2 = set()
        lits = [l for l in lits if not (l in seen2 or seen2.add(l))]
        return kws, lits

    @staticmethod
    def _score_chunk(chunk: Chunk, kws: List[str], lits: List[str]) -> float:
        text_low = chunk.text.lower()
        tokens = _tokenize(text_low)
        if not tokens:
            return 0.0
        token_set = set(tokens)
        # keyword overlap (IDF-free, but normalised by chunk size)
        kw_hits = sum(1 for k in kws if k in token_set)
        kw_score = kw_hits / (len(kws) ** 0.5 + 1e-6)

        # literal / unit presence
        lit_hits = sum(1 for l in lits if l and l in text_low)
        lit_score = 0.25 * lit_hits

        # numeric content boost
        nums = _NUMERIC_RE.findall(chunk.text)
        numeric_score = min(0.6, 0.05 * len(nums))

        # explicit unit presence
        unit_hits = sum(1 for u in _UNIT_TOKENS if u in token_set)
        unit_score = min(0.3, 0.05 * unit_hits)

        # section boost
        section = (chunk.section or "unknown").lower()
        sec_boost = 1.0
        for key, mult in _SECTION_BOOSTS.items():
            if section == key or section.startswith(key):
                sec_boost = mult
                break

        raw = kw_score + lit_score + numeric_score + unit_score
        return raw * sec_boost

    # ------------------------------------------------------------------
    # full per-document pipeline
    # ------------------------------------------------------------------

    def get_chunks_for_uri(self, uri: str) -> Tuple[List[Chunk], Dict[str, Any]]:
        """fetch -> download -> extract -> chunk. returns (chunks, info)."""
        info: Dict[str, Any] = {"uri": uri, "status": "ok"}
        work_id = _work_id_from_uri(uri)
        if not work_id:
            info["status"] = "not_openalex"
            return [], info
        info["work_id"] = work_id

        meta = self.fetch_metadata(work_id)
        if not meta:
            info["status"] = "meta_failed"
            return [], info

        candidates = self._candidate_pdf_urls(meta)
        info["n_candidates"] = len(candidates)
        if not candidates and not meta.get("doi"):
            info["status"] = "no_oa_pdf"
            return [], info

        pdf_path, dl_status = self.download_pdf(
            work_id, candidates, doi=meta.get("doi")
        )
        info["download_status"] = dl_status
        if not pdf_path:
            info["status"] = dl_status
            return [], info

        doc = self.extract_text(work_id, pdf_path)
        doc.source_pdf_url = candidates[0] if candidates else ""
        if not doc.extraction_ok:
            info["status"] = f"extract_failed:{doc.error or 'no_text'}"
            return [], info

        chunks = self.chunk_sections(doc)
        info["n_chunks"] = len(chunks)
        info["pages"] = doc.pages
        return chunks, info

    # ------------------------------------------------------------------
    # question-level selection
    # ------------------------------------------------------------------

    def select_excerpts_for_question(
        self,
        question: str,
        ontology: Any,
        documents: List[Dict[str, Any]],
        per_doc_budget: int = 6000,
        global_budget: int = 90000,
        top_k_per_doc: int = 10,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """score & select top chunks across all documents for this question.

        returns (excerpts, stats). each excerpt dict:
            {doc_index, title, work_id, section, page, text, tokens, score}
        documents: list of {title, abstract, uri} in the ORDER they appear in
                   the aql results (1-based doc_index follows that order).
        """
        kws, lits = self._build_query_terms(question, ontology)

        per_doc_chunks: List[List[Chunk]] = []
        per_doc_meta: List[Dict[str, Any]] = []

        for i, doc in enumerate(documents, start=1):
            uri = doc.get("uri") or ""
            chunks, info = self.get_chunks_for_uri(uri)
            info["doc_index"] = i
            info["title"] = doc.get("title", "")
            # score
            for c in chunks:
                c.score = self._score_chunk(c, kws, lits)
            chunks.sort(key=lambda c: c.score, reverse=True)

            # per-doc cap by top_k and per-doc token budget
            kept: List[Chunk] = []
            used_tok = 0
            for c in chunks[: top_k_per_doc * 3]:
                if len(kept) >= top_k_per_doc:
                    break
                if used_tok + c.tokens > per_doc_budget:
                    continue
                kept.append(c)
                used_tok += c.tokens
            per_doc_chunks.append(kept)
            info["kept"] = len(kept)
            info["kept_tokens"] = used_tok
            per_doc_meta.append(info)

        # global budget: flatten, sort by score desc, keep until global_budget
        all_candidates: List[Tuple[int, Chunk]] = []
        for i, chunks in enumerate(per_doc_chunks, start=1):
            for c in chunks:
                all_candidates.append((i, c))
        all_candidates.sort(key=lambda t: t[1].score, reverse=True)

        selected: List[Tuple[int, Chunk]] = []
        total = 0
        for doc_idx, c in all_candidates:
            if total + c.tokens > global_budget:
                continue
            selected.append((doc_idx, c))
            total += c.tokens

        # re-order selected: by doc_index then page for readable output
        selected.sort(key=lambda t: (t[0], t[1].page))

        excerpts: List[Dict[str, Any]] = []
        for doc_idx, c in selected:
            title = documents[doc_idx - 1].get("title", "")
            excerpts.append({
                "doc_index": doc_idx,
                "title": title,
                "work_id": c.work_id,
                "section": c.section,
                "page": c.page,
                "text": c.text,
                "tokens": c.tokens,
                "score": round(c.score, 3),
            })

        stats = {
            "n_docs": len(documents),
            "n_docs_with_chunks": sum(1 for c in per_doc_chunks if c),
            "n_excerpts": len(excerpts),
            "total_tokens": total,
            "per_doc": per_doc_meta,
        }
        return excerpts, stats

    # ------------------------------------------------------------------
    # rendering for prompt injection
    # ------------------------------------------------------------------

    @staticmethod
    def render_documents_block(
        full_docs: List[Dict[str, Any]],
        abstract_docs: List[Dict[str, Any]],
        excerpts: List[Dict[str, Any]],
        aql_lookup: Dict[str, Dict[str, Any]],
    ) -> str:
        """Render a single unified text block containing all documents (full +
        abstract) with their excerpts interleaved.

        full_docs come first (indices 1..len(full_docs)); abstract_docs follow.
        excerpts must carry a 'doc_index' key (1-based into full_docs) set by
        select_excerpts_for_question.
        aql_lookup maps URI -> metadata dict (title, abstract, year, authors).
        """
        from collections import defaultdict

        excerpts_by_idx: Dict[int, List[Dict]] = defaultdict(list)
        for ex in excerpts:
            excerpts_by_idx[ex.get("doc_index", -1)].append(ex)
        for idx in excerpts_by_idx:
            excerpts_by_idx[idx].sort(key=lambda e: e.get("page", 0))

        all_docs = list(full_docs) + list(abstract_docs)
        n_full   = len(full_docs)
        parts: List[str] = []

        for i, doc in enumerate(all_docs, start=1):
            uri  = doc.get("uri", "")
            meta = aql_lookup.get(uri, doc)

            title    = meta.get("title")    or doc.get("title")    or "Unknown"
            abstract = meta.get("abstract") or doc.get("abstract") or ""
            year     = meta.get("year")     or meta.get("publication_year") or ""
            authors  = meta.get("authors")  or meta.get("author")  or ""
            if isinstance(authors, list):
                authors = (
                    f"{authors[0]} et al." if len(authors) > 2
                    else ", ".join(str(a) for a in authors)
                )

            parts.append(f'=== Document [{i}]: "{title}" ===')
            parts.append(
                f"Authors: {authors} ({year}) | {uri}" if uri
                else f"Authors: {authors} ({year})"
            )
            parts.append(f"Abstract: {abstract}")

            doc_excerpts = excerpts_by_idx.get(i, []) if i <= n_full else []
            if doc_excerpts:
                parts.append("Full-text excerpts:")
                for ex in doc_excerpts:
                    parts.append(
                        f"  <<< Section: {ex.get('section', 'unknown')} "
                        f"| p. {ex.get('page', '?')} >>>"
                    )
                    parts.append(f"  {ex.get('text', '')}")
            else:
                parts.append(
                    "(Abstract only — no full-text selected for this document)"
                )

            parts.append("")  # blank separator between docs

        ref_lines: List[str] = []
        for i, doc in enumerate(all_docs, start=1):
            uri  = doc.get("uri", "")
            meta = aql_lookup.get(uri, doc)
            title   = meta.get("title")   or doc.get("title")   or "Unknown"
            year    = meta.get("year")    or meta.get("publication_year") or "n.d."
            authors = meta.get("authors") or meta.get("author")  or "Unknown"
            if isinstance(authors, list):
                authors = (
                    f"{authors[0]} et al." if len(authors) > 2
                    else ", ".join(str(a) for a in authors)
                )
            ref_lines.append(f"[{i}] {authors} ({year}). {title}. {uri}")

        parts.append("\n[VALIDATED REFERENCES]")
        parts.extend(ref_lines)

        return "\n".join(parts).strip()

    # ------------------------------------------------------------------
    # bulk prebuild (CLI)
    # ------------------------------------------------------------------

    def prebuild_from_csv(
        self, input_csv: str, limit_docs: Optional[int] = None,
        workers: int = 8,
    ) -> Dict[str, Any]:
        """walk the DLR CSV and prebuild metadata+PDF+text cache for every
        unique OpenAlex work that appears in aql_results.

        Runs across a thread pool (default 8 workers). Per-work_id cache files
        are disjoint, so there is no file contention. The OpenAlex API is
        polite-pooled via mailto and tolerates this fan-out. Playwright
        (if enabled) is internally serialized with a lock so threads will
        queue on the single headless Chromium instance."""
        csv.field_size_limit(int(1e8))
        uris: List[str] = []
        with open(input_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                raw = row.get("aql_results", "") or ""
                if not raw:
                    continue
                try:
                    docs = ast.literal_eval(raw)
                except Exception:
                    continue
                if not isinstance(docs, list):
                    continue
                for d in docs:
                    if isinstance(d, dict) and d.get("uri"):
                        uris.append(d["uri"])

        seen = set()
        unique_uris = [u for u in uris if not (u in seen or seen.add(u))]
        if limit_docs:
            unique_uris = unique_uris[:limit_docs]

        stats: Dict[str, Any] = {
            "total_unique": len(unique_uris),
            "meta_ok": 0, "pdf_ok": 0, "extract_ok": 0,
            "status_counts": {},
        }
        lock = threading.Lock()
        counters = {"done": 0, "ok": 0, "fail": 0}
        total = len(unique_uris)

        def _worker(uri: str) -> str:
            try:
                _, info = self.get_chunks_for_uri(uri)
                return info.get("status", "unknown")
            except Exception as exc:  # never let one bad doc kill the pool
                return f"worker_exception:{type(exc).__name__}"

        n_workers = max(1, int(workers))
        print(f"  [parallel] running with {n_workers} workers over {total} docs")

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_worker, u): u for u in unique_uris}
            for fut in as_completed(futures):
                s = fut.result()
                with lock:
                    stats["status_counts"][s] = stats["status_counts"].get(s, 0) + 1
                    if s == "ok":
                        stats["meta_ok"] += 1
                        stats["pdf_ok"] += 1
                        stats["extract_ok"] += 1
                        counters["ok"] += 1
                    else:
                        counters["fail"] += 1
                        if s not in ("meta_failed", "not_openalex"):
                            stats["meta_ok"] += 1
                        if s.startswith("extract_failed"):
                            stats["pdf_ok"] += 1
                    counters["done"] += 1
                    done = counters["done"]
                if done % 25 == 0 or done == total:
                    pct = (100 * counters["ok"] / done) if done else 0
                    print(
                        f"  [{done}/{total}] ok={counters['ok']} "
                        f"fail={counters['fail']} ({pct:.0f}% success)"
                    )
        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    ap = argparse.ArgumentParser(description="Full-text indexer for OpenAlex-backed RAG.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("prebuild", help="download + extract every OA pdf in the DLR csv")
    pb.add_argument("--input", required=True, help="DLR questions CSV")
    pb.add_argument("--limit", type=int, default=None, help="cap number of docs (for smoke test)")
    pb.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    pb.add_argument("--mailto", default=DEFAULT_MAILTO)
    pb.add_argument("--retry-failed", action="store_true",
                    help="delete .fail markers before running so previous failures are retried")
    pb.add_argument("--use-browser", action="store_true",
                    help="use Playwright (headless Chromium) as last-resort fallback for Cloudflare-gated publishers (Wiley, MDPI, OUP, IOP)")
    pb.add_argument("--workers", type=int, default=8,
                    help="parallel worker threads for prebuild (default 8). HTTP fetches run in parallel; Playwright is internally serialized.")

    st = sub.add_parser("status", help="print cache stats")
    st.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "prebuild":
        fti = FullTextIndexer(
            cache_dir=Path(args.cache_dir),
            mailto=args.mailto,
            use_browser=getattr(args, "use_browser", False),
        )
        if getattr(args, "retry_failed", False):
            fails = list((fti.pdf_dir).glob("*.fail"))
            for fp in fails:
                fp.unlink(missing_ok=True)
            print(f"cleared {len(fails)} .fail markers — will retry those docs")
        if fti.use_browser:
            print("browser fallback: ENABLED (Playwright Chromium will open lazily)")
        stats = fti.prebuild_from_csv(
            args.input, limit_docs=args.limit,
            workers=getattr(args, "workers", 8),
        )
        fti.close()
        print("\n=== prebuild complete ===")
        print(json.dumps(stats, indent=2))
    elif args.cmd == "status":
        base = Path(args.cache_dir)
        n_meta = len(list((base / "openalex").glob("*.json"))) if (base / "openalex").exists() else 0
        n_pdf = len(list((base / "pdfs").glob("*.pdf"))) if (base / "pdfs").exists() else 0
        n_text = len(list((base / "text").glob("*.json"))) if (base / "text").exists() else 0
        n_fail = len(list((base / "pdfs").glob("*.fail"))) if (base / "pdfs").exists() else 0
        print(json.dumps({
            "cache_dir": str(base),
            "meta_cached": n_meta,
            "pdfs_cached": n_pdf,
            "pdf_fail_markers": n_fail,
            "text_extracted": n_text,
        }, indent=2))


if __name__ == "__main__":
    _cli()
