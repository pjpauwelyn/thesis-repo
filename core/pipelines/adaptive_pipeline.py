"""backward-compat shim: import Pipeline from core.pipelines.pipeline."""
from core.pipelines.pipeline import Pipeline, PipelineResult  # noqa: F401

# keep old names working
AdaptivePipeline = Pipeline
AdaptiveResult   = PipelineResult
