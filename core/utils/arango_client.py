"""live ArangoDB KG client for the DLR EO knowledge graph.

drops-in as a replacement for reading pre-baked aql_results from the CSV:
call query_kg() with the LLM-rewritten search string and AQL params, get
back the same nested list[dict] structure that aql_parser.parse_aql_results
produces -- ready to pass straight into AdaptivePipeline.run().

required env vars (add to .env):
    ARANGO_URL            e.g. http://localhost:8529  or DLR hosted URL
    ARANGO_DB             database name, e.g. eo_rag
    ARANGO_ROOT_PASSWORD  ArangoDB root password
    ARANGO_USERNAME       optional, defaults to 'root'
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_AQL_PATH = Path(__file__).parent / "query.aql"

# fields to strip from raw ArangoDB docs before returning
_DROP_KEYS = {"_rev", "_id", "_key", "embedding"}


def _clean_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """strip internal ArangoDB fields recursively."""
    cleaned: Dict[str, Any] = {}
    for k, v in doc.items():
        if k in _DROP_KEYS or v is None:
            continue
        if isinstance(v, list):
            cleaned[k] = [_clean_doc(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def query_kg(
    search_query: str,
    kappa: int = 16,
    phi: int = 3,
    psi: int = 2,
    theta_expand: float = 0.3,
    aql_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """run the KG graph-expansion AQL and return cleaned doc list.

    parameters match DLR's InterfaceRAG._run_aql() signature so existing
    aql_params dicts (n/k/s/k_threshold) map directly.

    returns [] on connection failure so callers can fall back to CSV.
    """
    try:
        from arango import ArangoClient  # python-arango
    except ImportError as e:
        logger.error("python-arango not installed: %s  --  pip install python-arango", e)
        return []

    url      = os.getenv("ARANGO_URL", "http://localhost:8529")
    db_name  = os.getenv("ARANGO_DB", "eo_rag")
    username = os.getenv("ARANGO_USERNAME", "root")
    password = os.getenv("ARANGO_ROOT_PASSWORD", "")

    if not password:
        logger.warning("ARANGO_ROOT_PASSWORD not set; skipping live KG query")
        return []

    aql_file = aql_path or _AQL_PATH
    if not aql_file.exists():
        logger.error("AQL file not found: %s", aql_file)
        return []

    aql = aql_file.read_text()

    bind_vars = {
        "query":        search_query,
        "kappa":        kappa,
        "phi":          phi,
        "psi":          psi,
        "theta_expand": theta_expand,
    }

    try:
        client = ArangoClient(hosts=url)
        db = client.db(db_name, username=username, password=password)
        cursor = db.aql.execute(aql, bind_vars=bind_vars, count=True)
        results = [_clean_doc(doc) for doc in cursor]
        logger.info("KG query returned %d primary docs", len(results))
        return results
    except Exception as exc:
        logger.warning("KG query failed (%s); caller should fall back to CSV", exc)
        return []


def results_to_json(results: List[Dict[str, Any]]) -> str:
    """serialise KG results to the same compact JSON string that
    aql_parser.parse_aql_results() returns, so the rest of the pipeline
    is unchanged."""
    return json.dumps(results, separators=(",", ":"), ensure_ascii=False)
