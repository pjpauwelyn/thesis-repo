"""parse raw aql result strings into enriched json for the rag pipeline.

aql results arrive either as:
  - python repr strings (single-quoted dicts) from the DLR CSV column, or
  - already-clean JSON strings produced by arango_client.results_to_json()

previous version stripped everything down to title/abstract/uri (~90 % size
reduction) which threw away the entire KG dimension (science_keywords +
secondary_nodes).  this version keeps the full KG structure and only drops
internal ArangoDB bookkeeping fields and null values.
"""

import ast
import json
from typing import Any, Dict, List

# ArangoDB-internal fields that are never useful downstream
_DROP_KEYS = {"_rev", "_id", "_key", "embedding"}


def _clean_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """strip internal fields recursively (handles science_keywords lists)."""
    cleaned: Dict[str, Any] = {}
    for k, v in doc.items():
        if k in _DROP_KEYS or v is None:
            continue
        if isinstance(v, list):
            cleaned[k] = [_clean_doc(i) if isinstance(i, dict) else i for i in v]
        else:
            cleaned[k] = v
    return cleaned


def parse_aql_results(aql_results_str: str) -> str:
    """return compact JSON preserving the full KG structure per document.

    keeps: title, abstract, uri, science_keywords (name, description,
    secondary_nodes), and any other non-internal field present in the doc.
    drops: _id, _rev, _key, embedding, null values.

    accepts both Python repr strings (from CSV) and JSON strings
    (from arango_client.results_to_json).
    """
    if not aql_results_str or not aql_results_str.strip():
        return "[]"

    # try JSON first (arango_client path), then Python repr (CSV path)
    data: Any = None
    try:
        data = json.loads(aql_results_str)
    except (json.JSONDecodeError, ValueError):
        pass

    if data is None:
        try:
            data = ast.literal_eval(aql_results_str)
        except Exception as e:
            return json.dumps(
                {"error": f"aql parsing failed: {e}",
                 "original_length": len(aql_results_str)},
                separators=(",", ":"),
            )

    if not isinstance(data, list):
        return json.dumps([], separators=(",", ":"))

    clean_docs = [
        _clean_doc(doc)
        for doc in data
        if isinstance(doc, dict)
    ]
    return json.dumps(clean_docs, separators=(",", ":"), ensure_ascii=False)


def analyze_aql_results(aql_results_str: str) -> Dict[str, Any]:
    """return size and structure statistics for debugging / reporting."""
    original_size = len(aql_results_str)
    parsed = parse_aql_results(aql_results_str)
    parsed_size = len(parsed)

    try:
        data: Any = None
        try:
            data = json.loads(aql_results_str)
        except (json.JSONDecodeError, ValueError):
            data = ast.literal_eval(aql_results_str)

        doc_count = len(data) if isinstance(data, list) else 0
        kw_count = sum(
            len(d.get("science_keywords", []))
            for d in data
            if isinstance(d, dict)
        )
        sec_count = sum(
            len(sk.get("secondary_nodes", []))
            for d in data if isinstance(d, dict)
            for sk in d.get("science_keywords", [])
        )
        return {
            "original_size": original_size,
            "parsed_size": parsed_size,
            "reduction_pct": round((1 - parsed_size / original_size) * 100, 1) if original_size else 0,
            "document_count": doc_count,
            "science_keyword_count": kw_count,
            "secondary_node_count": sec_count,
        }
    except Exception:
        return {
            "original_size": original_size,
            "parsed_size": parsed_size,
            "error": "could not analyse original structure",
        }
