"""
Tests for adapters/assistant_api/run.py — streaming protocol compliance.

Verifies that _INTERNAL_TO_OPENAI_EVENT correctly maps every internal event
name to its OpenAI Assistants API SSE name (all outputs must start with
"thread.").
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.streaming]


class TestInternalToOpenAIEventMapping:
    """_INTERNAL_TO_OPENAI_EVENT must map every relevant internal name to its OpenAI SSE name."""

    @pytest.fixture(autouse=True)
    def _import_map(self):
        from llamphouse.core.adapters.assistant_api.run import _INTERNAL_TO_OPENAI_EVENT
        self.mapping = _INTERNAL_TO_OPENAI_EVENT

    def test_message_delta_maps_to_openai_name(self):
        assert self.mapping["message.delta"] == "thread.message.delta"

    def test_message_created_maps_to_openai_name(self):
        assert self.mapping["message.created"] == "thread.message.created"

    def test_message_in_progress_maps_to_openai_name(self):
        assert self.mapping["message.in_progress"] == "thread.message.in_progress"

    def test_message_completed_maps_to_openai_name(self):
        assert self.mapping["message.completed"] == "thread.message.completed"

    def test_run_created_maps_to_openai_name(self):
        assert self.mapping["run.created"] == "thread.run.created"

    def test_run_in_progress_maps_to_openai_name(self):
        assert self.mapping["run.in_progress"] == "thread.run.in_progress"

    def test_run_completed_maps_to_openai_name(self):
        assert self.mapping["run.completed"] == "thread.run.completed"

    def test_run_requires_action_maps_to_openai_name(self):
        # OpenAI uses 'requires_action'; internal name is 'awaiting_tools'
        assert self.mapping["run.awaiting_tools"] == "thread.run.requires_action"

    def test_run_failed_maps_to_openai_name(self):
        assert self.mapping["run.failed"] == "thread.run.failed"

    def test_run_step_created_maps_to_openai_name(self):
        assert self.mapping["run.step.created"] == "thread.run.step.created"

    def test_run_step_delta_maps_to_openai_name(self):
        assert self.mapping["run.step.delta"] == "thread.run.step.delta"

    def test_run_step_completed_maps_to_openai_name(self):
        assert self.mapping["run.step.completed"] == "thread.run.step.completed"

    def test_run_cancelled_maps_to_openai_name(self):
        assert self.mapping["run.cancelled"] == "thread.run.cancelled"

    def test_run_expired_maps_to_openai_name(self):
        assert self.mapping["run.expired"] == "thread.run.expired"

    def test_run_step_failed_maps_to_openai_name(self):
        assert self.mapping["run.step.failed"] == "thread.run.step.failed"

    def test_all_mapped_names_have_thread_prefix(self):
        """All OpenAI-facing event names must start with 'thread.'."""
        for internal, openai_name in self.mapping.items():
            assert openai_name.startswith("thread."), (
                f"Expected 'thread.' prefix for {internal!r} → {openai_name!r}"
            )
