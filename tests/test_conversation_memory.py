"""
tests/test_conversation_memory.py

Unit tests for ConversationMemoryTracker in src/models/conversation_memory.py.
Asserts strict temporal anti-leakage (preceding turns only), context bounds, and payload precedence.
"""

import pytest
from src.models.conversation_memory import ConversationMemoryTracker
from src.models.agent import TwitterSupportAgent


def test_single_turn_no_history():
    """Asserts single-turn messages with no preceding history return unadorned customer message."""
    tracker = ConversationMemoryTracker(max_turns=4, max_context_chars=2000)
    ctx = tracker.get_full_context(conversation_id="conv_1", current_message="Where is my order?")
    assert ctx == "Where is my order?"


def test_temporal_anti_leakage_t3_prediction():
    """
    CRITICAL METHODOLOGICAL TEST:
    Sequence:
    T1 Customer: Where is my package?
    T2 AmazonHelp: Let us check that for you.
    T3 Customer: Still waiting.  <-- PREDICTION POINT
    T4 AmazonHelp: Here is your tracking link: http://xyz
    T5 Customer: Thanks got it!

    Asserts prediction at T3 receives ONLY T1, T2, T3 and NEVER future turns T4, T5.
    """
    tracker = ConversationMemoryTracker(max_turns=4, max_context_chars=2000)
    conv_id = "conv_leak_test"

    # Add T1 and T2 to history
    tracker.add_turn(conv_id, "Customer", "Where is my package?")
    tracker.add_turn(conv_id, "AmazonHelp", "Let us check that for you.")

    # Request context at prediction point T3 ("Still waiting.")
    ctx = tracker.get_full_context(conv_id, current_message="Still waiting.")

    # Verification
    assert "Where is my package?" in ctx
    assert "Let us check that for you." in ctx
    assert "Still waiting." in ctx

    # ASSERT FUTURE TURNS ARE ABSENT
    assert "tracking link" not in ctx
    assert "Thanks got it!" not in ctx
    assert ctx.startswith("Preceding Context:")
    assert ctx.endswith("Current Inquiry:\nStill waiting.")


def test_max_turns_truncation():
    """Asserts that history is bounded to max_turns most recent preceding turns."""
    tracker = ConversationMemoryTracker(max_turns=2, max_context_chars=2000)
    conv_id = "conv_bound_test"

    tracker.add_turn(conv_id, "Customer", "Turn 1 customer")
    tracker.add_turn(conv_id, "AmazonHelp", "Turn 2 brand")
    tracker.add_turn(conv_id, "Customer", "Turn 3 customer")
    tracker.add_turn(conv_id, "AmazonHelp", "Turn 4 brand")

    ctx = tracker.get_full_context(conv_id, current_message="Turn 5 current inquiry")

    # Only Turn 3 and Turn 4 should be retained (max_turns=2)
    assert "Turn 1 customer" not in ctx
    assert "Turn 2 brand" not in ctx
    assert "Turn 3 customer" in ctx
    assert "Turn 4 brand" in ctx
    assert "Turn 5 current inquiry" in ctx


def test_custom_history_precedence():
    """Asserts custom_history list provided directly (e.g. from API payload) overrides server memory."""
    tracker = ConversationMemoryTracker(max_turns=4, max_context_chars=2000)
    conv_id = "conv_override"

    # Server memory has different content
    tracker.add_turn(conv_id, "Customer", "Server stored turn")

    # Client payload provides explicit history
    custom_history = ["Customer: Client turn 1", "AmazonHelp: Client reply 1"]
    ctx = tracker.get_full_context(conv_id, current_message="Client current inquiry", custom_history=custom_history)

    assert "Client turn 1" in ctx
    assert "Client reply 1" in ctx
    assert "Server stored turn" not in ctx


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def test_agent_multi_turn_processing():
    """Integration test verifying TwitterSupportAgent updates memory across multi-turn inquiries."""
    models_dir = BASE_DIR / "models"
    if (models_dir / "intent_tfidf_logreg.joblib").exists():
        agent = TwitterSupportAgent.load_from_models_dir(models_dir=models_dir, mock_llm=True)
    else:
        pytest.skip("Model artifacts missing in models/")

    conv_id = "conv_agent_test"

    res1 = agent.process_inquiry("Where is my package?", conversation_id=conv_id)
    assert res1["conversation_id"] == conv_id
    assert res1["query_context_used"] == "Where is my package?"

    # Turn 2: Follow-up message
    res2 = agent.process_inquiry("Still waiting.", conversation_id=conv_id)
    assert "Where is my package?" in res2["query_context_used"]
    assert "Still waiting." in res2["query_context_used"]
