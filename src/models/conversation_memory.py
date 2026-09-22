"""
src/models/conversation_memory.py

Implements ConversationMemoryTracker for stateful multi-turn customer support context tracking.
Strictly enforces temporal anti-leakage (preceding turns only) and bounded context limits.
"""

import time
from typing import Dict, List, Optional


class ConversationMemoryTracker:
    """
    Manages preceding conversation turns per conversation_id.
    Prevents temporal leakage by strictly tracking history prior to the current inquiry.
    Applies bounds on turns and total character count.
    """

    def __init__(self, max_turns: int = 4, max_context_chars: int = 2000):
        self.max_turns = max_turns
        self.max_context_chars = max_context_chars
        # Store histories: Dict[conversation_id, List[Dict[str, Any]]]
        self._store: Dict[str, List[Dict[str, str]]] = {}

    def add_turn(self, conversation_id: str, role: str, text: str) -> None:
        """
        Adds a single turn (role: 'Customer' or 'AmazonHelp') to memory.
        """
        if not conversation_id or not text:
            return
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        
        self._store[conversation_id].append({
            "role": role,
            "text": text.strip()
        })

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Returns the list of stored turns for a given conversation_id."""
        return list(self._store.get(conversation_id, []))

    def clear(self, conversation_id: Optional[str] = None) -> None:
        """Clears memory for a specific conversation_id or all conversations."""
        if conversation_id is not None:
            self._store.pop(conversation_id, None)
        else:
            self._store.clear()

    def format_history_lines(self, history_items: List[Dict[str, str]]) -> List[str]:
        """
        Formats structured history items into strings.
        Supports both dict format ({'role': ..., 'text': ...}) and raw string lines.
        """
        formatted = []
        for item in history_items:
            if isinstance(item, dict):
                role = item.get("role", "Customer")
                text = item.get("text") or item.get("content", "")
                formatted.append(f"{role}: {text}")
            elif isinstance(item, str):
                formatted.append(item.strip())
        return formatted

    def get_full_context(
        self,
        conversation_id: Optional[str],
        current_message: str,
        custom_history: Optional[List[str]] = None
    ) -> str:
        """
        Constructs the augmented prompt context for intent classification & retrieval.
        
        Precedence:
        1. If custom_history is provided (e.g. from API payload), it is used as preceding context.
        2. Otherwise, if conversation_id exists in self._store, stored history is used.
        3. If no preceding context exists, current_message is returned as-is.

        Strict Temporal & Bound Constraints:
        - Only preceding turns are included.
        - Truncated to self.max_turns most recent preceding turns.
        - Truncated to self.max_context_chars maximum length.
        """
        current_clean = current_message.strip()
        
        # Determine source history
        raw_history: List[str] = []
        if custom_history is not None and len(custom_history) > 0:
            raw_history = self.format_history_lines(custom_history)
        elif conversation_id and conversation_id in self._store:
            raw_history = self.format_history_lines(self._store[conversation_id])
            
        if not raw_history:
            return current_clean

        # Enforce max_turns bound (retain most recent preceding turns)
        bounded_history = raw_history[-self.max_turns:]
        
        context_block = "\n".join(bounded_history)
        
        # Enforce character bound if context block exceeds max_context_chars
        if len(context_block) > self.max_context_chars:
            context_block = context_block[-self.max_context_chars:]
            
        return f"Preceding Context:\n{context_block}\n\nCurrent Inquiry:\n{current_clean}"
