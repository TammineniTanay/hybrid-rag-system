import pytest
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from batch_eval import questions

class TestBatchEvaluator:
    """Tests for the batch evaluation pipeline."""

    def test_questions_list_not_empty(self):
        """Ensure question bank is populated."""
        assert len(questions) > 0

    def test_questions_are_strings(self):
        """All questions must be non-empty strings."""
        for q in questions:
            assert isinstance(q, str)
            assert len(q.strip()) > 0

    def test_questions_count(self):
        """Verify we have exactly 50 evaluation questions."""
        assert len(questions) == 50

    def test_eval_results_file_exists(self):
        """Check that evaluation results file was generated."""
        assert os.path.exists("eval_results_50.json")

    def test_eval_results_valid_json(self):
        """Results file must be valid JSON."""
        with open("eval_results_50.json", "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_eval_results_have_latency(self):
        """Every result must have a latency field."""
        with open("eval_results_50.json", "r") as f:
            data = json.load(f)
        for result in data:
            assert "latency_s" in result

    def test_eval_results_have_questions(self):
        """Every result must reference the original question."""
        with open("eval_results_50.json", "r") as f:
            data = json.load(f)
        for result in data:
            assert "question" in result or "error" in result

    def test_average_latency_reasonable(self):
        """Average latency should be under 120 seconds per query."""
        with open("eval_results_50.json", "r") as f:
            data = json.load(f)
        latencies = [r["latency_s"] for r in data if "latency_s" in r]
        if not latencies:
            pytest.skip("No latency data available")
        avg = sum(latencies) / len(latencies)
        # Real evaluation showed avg ~67s which is acceptable for a RAG pipeline
        # querying multiple vector stores and an LLM
        assert avg < 120, f"Average latency {avg:.1f}s exceeds threshold"