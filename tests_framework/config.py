"""
test configuration: questions, paths, thresholds.

everything a test run needs in one place.  all paths are relative to the
project root (the directory containing `core/`).
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# project root – walk up from this file to find the directory with core/
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent  # refactored/


# ---------------------------------------------------------------------------
# representative test questions (earth observation domain)
# kept small (1-3) so the suite runs in < 5 minutes on a laptop
# ---------------------------------------------------------------------------

TEST_QUESTIONS: List[Dict[str, str]] = [
    {
        "question_id": "901",
        "question": (
            "What satellite missions are commonly used for monitoring "
            "global sea surface temperature, and what spatial resolution "
            "do they typically achieve?"
        ),
        # minimal structured context that the refinement agent can work with
        "structured_context": (
            "## RELATED DOCUMENTS\n"
            "---\n"
            "# Title: Sentinel-3 Sea Surface Temperature Monitoring\n"
            "# Content: The Sentinel-3 mission carries the SLSTR instrument "
            "which measures sea surface temperature at 1 km resolution. "
            "MODIS on Aqua and Terra also provides SST data at similar "
            "resolutions. These missions enable global ocean monitoring "
            "for climate research and weather forecasting applications.\n"
            "---\n"
            "# Title: VIIRS SST Products from Suomi NPP\n"
            "# Content: The VIIRS instrument on Suomi NPP satellite provides "
            "sea surface temperature products at 750 m resolution. The data "
            "is used in operational oceanography and has demonstrated accuracy "
            "comparable to in-situ measurements from drifting buoys.\n"
        ),
        # synthetic aql results (compact json the pipeline normally receives)
        "aql_results": (
            '[{"title":"Sentinel-3 SST","abstract":"Sentinel-3 SLSTR '
            'provides 1km SST data for climate monitoring.",'
            '"uri":"https://openalex.org/W0000000001"},'
            '{"title":"VIIRS SST Products","abstract":"VIIRS on Suomi NPP '
            'delivers 750m resolution SST.","uri":"https://openalex.org/W0000000002"}]'
        ),
    },
    {
        "question_id": "902",
        "question": (
            "How is synthetic aperture radar (SAR) used for flood mapping, "
            "and what are the main limitations?"
        ),
        "structured_context": (
            "## RELATED DOCUMENTS\n"
            "---\n"
            "# Title: SAR-Based Flood Detection Methods\n"
            "# Content: Synthetic aperture radar is widely used for flood "
            "mapping due to its ability to penetrate clouds. Sentinel-1 C-band "
            "SAR data is commonly used. Key limitations include double-bounce "
            "effects in urban areas and difficulty distinguishing flooded "
            "vegetation from open water.\n"
        ),
        "aql_results": (
            '[{"title":"SAR Flood Detection","abstract":"SAR provides '
            'all-weather flood mapping capability using backscatter analysis.",'
            '"uri":"https://openalex.org/W0000000003"}]'
        ),
    },
]

# a third question intentionally has empty context to exercise the
# "no documents available" fallback path
TEST_QUESTIONS_WITH_EMPTY = TEST_QUESTIONS + [
    {
        "question_id": "903",
        "question": "What is the latest status of the NISAR satellite mission?",
        "structured_context": "",
        "aql_results": "",
    },
]


# ---------------------------------------------------------------------------
# output paths (inside a test-specific temp directory by default)
# ---------------------------------------------------------------------------

@dataclass
class TestPaths:
    """all output locations for a single test run."""
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "test_output")
    pipeline_csv: Path = field(init=False)
    eval_csv: Path = field(init=False)
    summary_json: Path = field(init=False)
    summary_csv: Path = field(init=False)
    log_file: Path = field(init=False)

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_csv = self.output_dir / "test_pipeline_results.csv"
        self.eval_csv = self.output_dir / "test_evaluation_results.csv"
        self.summary_json = self.output_dir / "test_summary.json"
        self.summary_csv = self.output_dir / "test_summary.csv"
        self.log_file = self.output_dir / "test_run.log"


# ---------------------------------------------------------------------------
# sanity-check thresholds
# ---------------------------------------------------------------------------

@dataclass
class Thresholds:
    # pipeline output
    min_answer_length: int = 30          # chars; anything shorter is suspect
    min_context_docs: int = 1            # at least 1 doc in context
    max_processing_time: float = 120.0   # seconds per question

    # evaluation output
    valid_score_range: tuple = (1, 5)    # dlr criteria scores
    min_overall_score: float = 1.0       # -1 means failure
    required_criteria: tuple = (
        "factuality", "relevance", "groundedness", "helpfulness", "depth",
    )


# ---------------------------------------------------------------------------
# failure markers – strings that indicate a known error in logs or outputs
# ---------------------------------------------------------------------------

FAILURE_MARKERS = [
    "No documents available",
    "Could not generate answer",
    "Error:",
    "ERROR:",
    "error generating refined context",
    "empty LLM response",
    "Traceback (most recent call last)",
    "Exception",
]

# patterns that are fine even though they contain "Error" as a substring
FAILURE_MARKER_EXCEPTIONS = [
    "error_message",  # csv header
    "error_msg",      # analytics field
]
