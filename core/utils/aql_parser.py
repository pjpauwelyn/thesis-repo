"""parse raw aql result strings into compact json for the rag pipeline.

aql results arrive as python repr strings (single-quoted dicts inside a list).
this module extracts only the fields the pipeline needs: title, abstract, uri.
"""

import ast
import json
from typing import List, Dict, Any


def parse_aql_results(aql_results_str: str) -> str:
    """return compact json containing only title/abstract/uri per document.

    expected input: python repr string from csv.
    expected size reduction: ~90 % per question.
    """
    if not aql_results_str or not aql_results_str.strip():
        return "[]"

    try:
        data = ast.literal_eval(aql_results_str)
        if not isinstance(data, list):
            return json.dumps([], separators=(",", ":"))

        clean_docs = [
            {
                "title": doc.get("title", ""),
                "abstract": doc.get("abstract", ""),
                "uri": doc.get("uri", ""),
            }
            for doc in data
            if isinstance(doc, dict)
        ]
        return json.dumps(clean_docs, separators=(",", ":"))

    except Exception as e:
        return json.dumps(
            {"error": f"aql parsing failed: {e}", "original_length": len(aql_results_str)},
            separators=(",", ":"),
        )


def analyze_aql_results(aql_results_str: str) -> Dict[str, Any]:
    """return size-reduction statistics for debugging / reporting."""
    original_size = len(aql_results_str)
    parsed = parse_aql_results(aql_results_str)
    parsed_size = len(parsed)

    try:
        data = ast.literal_eval(aql_results_str)
        doc_count = len(data) if isinstance(data, list) else 0
        content_chars = sum(
            len(d.get("title", "")) + len(d.get("abstract", "")) + len(d.get("uri", ""))
            for d in data
            if isinstance(d, dict)
        )
        return {
            "original_size": original_size,
            "parsed_size": parsed_size,
            "reduction_pct": round((1 - parsed_size / original_size) * 100, 1) if original_size else 0,
            "document_count": doc_count,
            "content_chars": content_chars,
        }
    except Exception:
        return {
            "original_size": original_size,
            "parsed_size": parsed_size,
            "error": "could not analyse original structure",
        }
