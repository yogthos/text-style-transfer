"""Linguistic quality integration tests.

Tests for zipper merge, action echo, grounding validation, and perspective lock.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.utils.text_processing import check_zipper_merge
from src.validator.statistical_critic import StatisticalCritic
from src.validator.semantic_critic import SemanticCritic
from src.generator.translator import StyleTranslator
from tests.test_helpers import ensure_config_exists

# Ensure config exists
ensure_config_exists()


class TestAntiStutterZipperMerge:
    """Test anti-stutter (zipper merge) detection."""

    def test_full_echo_detection(self):
        """Test that full echo is detected."""
        prev_sent = "The doors opened."
        new_sent = "The doors opened again."

        result = check_zipper_merge(prev_sent, new_sent)
        assert result is True, "Full echo should be detected"

    def test_head_echo_detection(self):
        """Test that head echo (same 3+ words at start) is detected."""
        prev_sent = "I walked home."
        new_sent = "I walked to the store."

        result = check_zipper_merge(prev_sent, new_sent)
        assert result is True, "Head echo should be detected"

    def test_tail_echo_detection(self):
        """Test that tail echo (end of prev matches start of new) is detected."""
        prev_sent = "The doors opened."
        new_sent = "Opened, the room revealed its secrets."

        result = check_zipper_merge(prev_sent, new_sent)
        # Note: This may not always detect tail echo depending on implementation
        # The check looks for first 3 words of new_sent in last 6 words of prev_sent
        # "Opened" appears in "doors opened" so it should be detected
        assert result is True, "Tail echo should be detected"

    def test_no_echo_passes(self):
        """Test that non-echoing sentences pass."""
        prev_sent = "The doors opened."
        new_sent = "I entered the room."

        result = check_zipper_merge(prev_sent, new_sent)
        assert result is False, "Non-echoing sentences should pass"


class TestActionEchoDetection:
    """Test action echo detection using spaCy lemmatization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.critic = StatisticalCritic(config_path="config.json")

    def test_action_echo_detected_weave(self):
        """Test that 'weaving' and 'wove' are detected as same action."""
        sentences = [
            "I was weaving a basket.",
            "I wove another one."
        ]

        issues = self.critic.check_action_echo(sentences)
        assert len(issues) > 0, "Action echo should be detected (weave/wove)"
        assert any("weave" in issue.lower() or "wove" in issue.lower() for issue in issues)

    def test_action_echo_detected_run(self):
        """Test that 'ran' and 'runs' are detected as same action."""
        sentences = [
            "She ran fast.",
            "He runs daily."
        ]

        issues = self.critic.check_action_echo(sentences)
        assert len(issues) > 0, "Action echo should be detected (run/ran/runs)"
        assert any("run" in issue.lower() for issue in issues)

    def test_auxiliary_verbs_ignored(self):
        """Test that auxiliary verbs (was, had, did) are ignored."""
        sentences = [
            "I was happy.",
            "She was sad."
        ]

        issues = self.critic.check_action_echo(sentences)
        # Should NOT detect echo for auxiliaries
        assert len(issues) == 0, "Auxiliary verbs should not trigger action echo"

    def test_no_action_echo_passes(self):
        """Test that different actions pass."""
        sentences = [
            "I walked home.",
            "She drove to work."
        ]

        issues = self.critic.check_action_echo(sentences)
        assert len(issues) == 0, "Different actions should not trigger echo"


class TestGroundingValidation:
    """Test grounding validation (anti-moralizing)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.critic = SemanticCritic(config_path="config.json")

    def test_abstract_moralizing_fails(self):
        """Test that abstract/moralizing endings fail."""
        paragraph = "This is a paragraph about something. Thus, I learned about society."

        issue = self.critic.check_ending_grounding(paragraph)
        assert issue is not None, "Abstract/moralizing ending should fail"
        assert "abstract" in issue.lower() or "moralizing" in issue.lower()

    def test_concrete_detail_passes(self):
        """Test that concrete sensory details pass."""
        paragraph = "This is a paragraph about something. The door closed with a click."

        issue = self.critic.check_ending_grounding(paragraph)
        assert issue is None, "Concrete detail should pass"

    def test_moralizing_pattern_fails(self):
        """Test that moralizing patterns fail."""
        paragraph = "This is a paragraph. In conclusion, the lesson is clear."

        issue = self.critic.check_ending_grounding(paragraph)
        assert issue is not None, "Moralizing pattern should fail"


class TestPerspectiveLock:
    """Test perspective locking verification."""

    def setup_method(self):
        """Set up test fixtures."""
        ensure_config_exists()
        self.translator = StyleTranslator(config_path="config.json")

    def test_first_person_singular_lock(self):
        """Test that first person singular input locks perspective."""
        # Test with text that should maintain first person
        generated_text = "I went to the store. I bought some food. I returned home."

        result = self.translator.verify_perspective(generated_text, "first_person_singular")
        assert result is True, "First person singular should pass"

    def test_first_person_singular_fails_with_third(self):
        """Test that first person fails when third person pronouns appear."""
        generated_text = "I went to the store. He bought some food."

        result = self.translator.verify_perspective(generated_text, "first_person_singular")
        assert result is False, "First person should fail when third person appears"

    def test_third_person_lock(self):
        """Test that third person input locks perspective."""
        generated_text = "He walked home. She drove to work. They met at the cafe."

        result = self.translator.verify_perspective(generated_text, "third_person")
        assert result is True, "Third person should pass"

    def test_third_person_fails_with_first(self):
        """Test that third person fails when first person pronouns appear."""
        generated_text = "He walked home. I drove to work."

        result = self.translator.verify_perspective(generated_text, "third_person")
        assert result is False, "Third person should fail when first person appears"

    def test_first_person_plural_lock(self):
        """Test that first person plural input locks perspective."""
        generated_text = "We went to the store. We bought some food. We returned home."

        result = self.translator.verify_perspective(generated_text, "first_person_plural")
        assert result is True, "First person plural should pass"

    def test_empty_text_passes(self):
        """Test that empty text passes (no pronouns to check)."""
        result = self.translator.verify_perspective("", "first_person_singular")
        assert result is True, "Empty text should pass"

