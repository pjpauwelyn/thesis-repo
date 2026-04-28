"""fetch publication metadata from the openalex api."""

import logging
import time
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

_OPENALEX_BASE = "https://api.openalex.org/works/"
_MAX_RETRIES = 2


class OpenAlexClient:
    """stateless client with module-level URI cache."""

    _cache: Dict[str, Optional[Dict[str, str]]] = {}  # module-level, persists per process

    @classmethod
    def fetch_metadata(cls, uri: str) -> Optional[Dict[str, str]]:
        """return {author, year, title, uri} or None on failure."""
        if not uri or "openalex.org" not in uri:
            return None

        # Return cached result immediately (None means previously failed)
        if uri in cls._cache:
            return cls._cache[uri]

        work_id = uri.rstrip("/").split("/")[-1]
        if not work_id:
            cls._cache[uri] = None
            return None

        delay = 1.0
        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(f"{_OPENALEX_BASE}{work_id}", timeout=3)
                resp.raise_for_status()
                data = resp.json()

                author = "No author"
                authorships = data.get("authorships") or []
                if authorships:
                    author = authorships[0].get("author", {}).get("display_name", "No author")

                year = data.get("publication_year")
                result = {
                    "author": author,
                    "year": str(year) if year else "No year",
                    "title": data.get("title") or "",
                    "uri": uri,
                }
                cls._cache[uri] = result
                return result
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    cls._cache[uri] = None
                    return None
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning(f"openalex fetch failed for {uri}: {exc}")
                    cls._cache[uri] = None
                    return None
            except requests.exceptions.RequestException as exc:
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning(f"openalex fetch failed for {uri}: {exc}")
                    cls._cache[uri] = None
                    return None
            except Exception as exc:
                logger.error(f"unexpected error fetching {uri}: {exc}")
                cls._cache[uri] = None
                return None


def format_reference_from_metadata(metadata: Dict[str, str]) -> str:
    """format a numbered reference line from openalex metadata.

    P5: title is normalised to empty string by fetch_metadata when OpenAlex
    returns null or "".  The caller (_build_verified_references) checks for
    empty title and falls back to the KG title before calling this function,
    so an empty title here should be rare.  As a belt-and-suspenders guard,
    replace empty title with a placeholder that is clearly not a data value.
    """
    pos = metadata.get("position", 1)
    author = metadata.get("author") or "No author"
    year = metadata.get("year") or "No year"
    title = (metadata.get("title") or "").strip() or "[No title \u2014 see URI]"
    uri = metadata.get("uri", "")
    return f"[{pos}] {author} ({year}). *{title}*. {uri}."
