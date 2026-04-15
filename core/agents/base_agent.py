"""abstract base for all pipeline agents."""

import time
from abc import ABC, abstractmethod
from typing import Any

from core.utils.helpers import setup_logging


class BaseAgent(ABC):
    def __init__(self, name: str, llm):
        self.name = name
        self.llm = llm
        self.logger = setup_logging(self.name)

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        ...

    def run(self, input_data: Any) -> Any:
        """execute process() with timing and error logging."""
        start = time.time()
        self.logger.info(f"starting {self.name}")
        try:
            output = self.process(input_data)
            self.logger.info(f"completed in {time.time() - start:.2f}s")
            return output
        except Exception as e:
            self.logger.error(f"error: {e}", exc_info=True)
            raise
