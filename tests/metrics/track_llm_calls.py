"""LLM call tracking for efficiency regression detection.

Tracks LLM call counts and patterns to detect performance regressions.
"""

import json
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class LLMCallTracker:
    """Tracks LLM calls for performance monitoring."""

    def __init__(self, metrics_dir: Optional[Path] = None):
        """Initialize tracker.

        Args:
            metrics_dir: Directory to store metrics files. Defaults to tests/metrics/
        """
        if metrics_dir is None:
            metrics_dir = Path(__file__).parent
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.call_history: List[Dict[str, Any]] = []
        self.call_count = 0
        self.paragraph_count = 0

    def record_call(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        model_type: str = "editor",
        response_length: int = 0
    ):
        """Record an LLM call.

        Args:
            system_prompt: System prompt (truncated for storage)
            user_prompt: User prompt (truncated for storage)
            model_type: Model type used
            response_length: Length of response in characters
        """
        self.call_count += 1
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "call_number": self.call_count,
            "system_prompt": system_prompt[:100] if len(system_prompt) > 100 else system_prompt,
            "user_prompt": user_prompt[:200] if len(user_prompt) > 200 else user_prompt,
            "model_type": model_type,
            "response_length": response_length
        })

    def record_paragraph(self, paragraph_id: str, call_count: int):
        """Record LLM calls for a paragraph.

        Args:
            paragraph_id: Identifier for the paragraph
            call_count: Number of LLM calls made for this paragraph
        """
        self.paragraph_count += 1
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "paragraph",
            "paragraph_id": paragraph_id,
            "call_count": call_count,
            "paragraph_number": self.paragraph_count
        })

    def save_history(self, filename: str = "llm_call_history.json"):
        """Save call history to file.

        Args:
            filename: Name of file to save
        """
        filepath = self.metrics_dir / filename
        with open(filepath, 'w') as f:
            json.dump({
                "total_calls": self.call_count,
                "total_paragraphs": self.paragraph_count,
                "calls_per_paragraph": self.call_count / self.paragraph_count if self.paragraph_count > 0 else 0,
                "history": self.call_history
            }, f, indent=2)

    def load_baseline(self, filename: str = "baseline_metrics.json") -> Dict[str, Any]:
        """Load baseline metrics for comparison.

        Args:
            filename: Name of baseline file

        Returns:
            Baseline metrics dictionary
        """
        filepath = self.metrics_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}

    def compare_to_baseline(self, threshold: float = 0.5) -> tuple[bool, Dict[str, Any]]:
        """Compare current metrics to baseline.

        Args:
            threshold: Maximum allowed increase (0.5 = 50% increase)

        Returns:
            Tuple of (passes, comparison_dict)
        """
        baseline = self.load_baseline()

        if not baseline:
            # No baseline, create one
            self.save_baseline()
            return True, {"status": "baseline_created"}

        baseline_calls_per_para = baseline.get("calls_per_paragraph", 0)
        current_calls_per_para = self.call_count / self.paragraph_count if self.paragraph_count > 0 else 0

        if baseline_calls_per_para == 0:
            return True, {"status": "no_baseline_data"}

        increase = (current_calls_per_para - baseline_calls_per_para) / baseline_calls_per_para
        passes = increase <= threshold

        comparison = {
            "passes": passes,
            "baseline_calls_per_paragraph": baseline_calls_per_para,
            "current_calls_per_paragraph": current_calls_per_para,
            "increase_percentage": increase * 100,
            "threshold_percentage": threshold * 100
        }

        return passes, comparison

    def save_baseline(self, filename: str = "baseline_metrics.json"):
        """Save current metrics as baseline.

        Args:
            filename: Name of baseline file
        """
        filepath = self.metrics_dir / filename
        with open(filepath, 'w') as f:
            json.dump({
                "total_calls": self.call_count,
                "total_paragraphs": self.paragraph_count,
                "calls_per_paragraph": self.call_count / self.paragraph_count if self.paragraph_count > 0 else 0,
                "created_at": datetime.now().isoformat()
            }, f, indent=2)


# Global tracker instance
_tracker: Optional[LLMCallTracker] = None


def get_tracker() -> LLMCallTracker:
    """Get global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = LLMCallTracker()
    return _tracker


def count_llm_calls(func):
    """Decorator to count LLM invocations.

    Works with both real and mocked LLM providers.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracker = get_tracker()

        # Call original function
        result = func(*args, **kwargs)

        # Record call
        tracker.record_call(
            system_prompt=str(kwargs.get("system_prompt", ""))[:100],
            user_prompt=str(kwargs.get("user_prompt", ""))[:200],
            model_type=str(kwargs.get("model_type", "editor")),
            response_length=len(str(result)) if result else 0
        )

        return result

    return wrapper

