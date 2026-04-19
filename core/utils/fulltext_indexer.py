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
import time
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
    ):
        self.cache_dir = Path(cache_dir)
        self.mailto = mailto
        self.request_timeout = request_timeout
        self.max_pdf_bytes = max_pdf_mb * 1024 * 1024

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
        """parse <meta name='citation_pdf_url'> or ...fulltext_url from HTML."""
        if not html_text:
            return None
        for rx in (_CITATION_PDF_META_RE, _CITATION_PDF_FULLTEXT_RE):
            m = rx.search(html_text)
            if m:
                url = m.group(1).strip()
                if url.startswith("//"):
                    url = "https:" + url
                elif url.startswith("/") and "://" in base_url:
                    root = "/".join(base_url.split("/")[:3])
                    url = root + url
                return url
        return None

    # ------------------------------------------------------------------
    # pdf download
    # ------------------------------------------------------------------

    def _try_fetch_pdf(self, url: str, cache_path: Path) -> Tuple[bool, str]:
        """attempt to fetch ONE candidate URL. returns (success, reason).

        if the response is HTML, parses <meta citation_pdf_url> and recurses
        ONCE. keeps no fail marker here; caller decides."""
        try:
            with self.session.get(
                url, timeout=self.request_timeout, stream=True, allow_redirects=True
            ) as resp:
                if resp.status_code >= 400:
                    return False, f"http_{resp.status_code}"
                ctype = (resp.headers.get("content-type") or "").lower()
                final_url = resp.url or url
                # HTML: try to resolve to a real PDF URL
                if "pdf" not in ctype and not final_url.lower().endswith(".pdf"):
                    # only look at first ~500KB of HTML
                    if "text/html" in ctype or "xml" in ctype:
                        body = resp.raw.read(512 * 1024, decode_content=True) or b""
                        try:
                            html_text = body.decode("utf-8", errors="ignore")
                        except Exception:
                            html_text = ""
                        resolved = self._resolve_pdf_from_html(html_text, final_url)
                        if resolved and resolved != url:
                            # one-level recursion with fresh request
                            return self._try_fetch_pdf(resolved, cache_path)
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
        for url in candidates:
            if not url:
                continue
            ok, why = self._try_fetch_pdf(url, cache_path)
            if ok:
                return cache_path, "ok"
            reasons.append(f"{url[:60]}->{why}")

        # Unpaywall fallback
        unpaywall_url = self._unpaywall_pdf_url(doi)
        if unpaywall_url and unpaywall_url not in candidates:
            ok, why = self._try_fetch_pdf(unpaywall_url, cache_path)
            if ok:
                return cache_path, "ok_unpaywall"
            reasons.append(f"unpaywall:{why}")

        fail_marker.write_text(" | ".join(reasons)[:400] or "no_candidates")
        return None, "pdf_download_failed"

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
    def render_excerpts_block(excerpts: List[Dict[str, Any]]) -> str:
        if not excerpts:
            return "No full-text excerpts available for this question (abstracts only)."
        parts: List[str] = []
        current_doc = None
        for ex in excerpts:
            di = ex["doc_index"]
            if di != current_doc:
                parts.append(f"\n### Document [{di}] — {ex.get('title','')}")
                current_doc = di
            parts.append(
                f"<<< Doc [{di}] | Section: {ex.get('section','unknown')} | p. {ex.get('page','?')} >>>\n"
                f"{ex.get('text','')}"
            )
        return "\n".join(parts).strip()

    # ------------------------------------------------------------------
    # bulk prebuild (CLI)
    # ------------------------------------------------------------------

    def prebuild_from_csv(
        self, input_csv: str, limit_docs: Optional[int] = None
    ) -> Dict[str, Any]:
        """walk the DLR CSV and prebuild metadata+PDF+text cache for every
        unique OpenAlex work that appears in aql_results."""
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

        stats = {
            "total_unique": len(unique_uris),
            "meta_ok": 0, "pdf_ok": 0, "extract_ok": 0,
            "status_counts": {},
        }
        running_ok = 0
        running_fail = 0
        for i, uri in enumerate(unique_uris, 1):
            _, info = self.get_chunks_for_uri(uri)
            s = info.get("status", "unknown")
            stats["status_counts"][s] = stats["status_counts"].get(s, 0) + 1
            if s == "ok":
                stats["meta_ok"] += 1
                stats["pdf_ok"] += 1
                stats["extract_ok"] += 1
                running_ok += 1
            else:
                running_fail += 1
                if s != "meta_failed" and s != "not_openalex":
                    stats["meta_ok"] += 1
                if s in ("extract_failed",) or s.startswith("extract_failed"):
                    stats["pdf_ok"] += 1
            if i % 25 == 0 or i == len(unique_uris):
                total = i
                print(
                    f"  [{i}/{len(unique_uris)}] ok={running_ok} "
                    f"fail={running_fail} ({100*running_ok/total:.0f}% success)"
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

    st = sub.add_parser("status", help="print cache stats")
    st.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.cmd == "prebuild":
        fti = FullTextIndexer(cache_dir=Path(args.cache_dir), mailto=args.mailto)
        if getattr(args, "retry_failed", False):
            fails = list((fti.pdf_dir).glob("*.fail"))
            for fp in fails:
                fp.unlink(missing_ok=True)
            print(f"cleared {len(fails)} .fail markers — will retry those docs")
        stats = fti.prebuild_from_csv(args.input, limit_docs=args.limit)
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
