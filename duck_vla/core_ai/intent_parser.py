"""
Language module: Intent parser for command processing

This module extracts structured intents from natural language commands.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


class IntentParser:
    """
    Parse natural language commands into structured intents.

    This is a rule-based intent parser that maps phrases to action intents
    using pattern matching.
    """

    def __init__(self):
        """Initialize the intent parser with command patterns."""
        logger.info("Initializing intent parser")

        # Define command patterns for different actions
        self.patterns = {
            # Movement commands
            "move_forward": [
                r"(?:go|move|walk)(?:\s+(?:forward|ahead))?",
                r"forward",
                r"ahead",
            ],
            "move_backward": [
                r"(?:go|move|walk)\s+(?:back|backward)",
                r"backwards?",
                r"back(?:\s+up)?",
            ],
            "turn_left": [
                r"turn\s+(?:to\s+)?(?:the\s+)?left",
                r"(?:go|move|step)\s+left",
                r"left",
            ],
            "turn_right": [
                r"turn\s+(?:to\s+)?(?:the\s+)?right",
                r"(?:go|move|step)\s+right",
                r"right",
            ],
            "turn_around": [
                r"turn\s+around",
                r"(?:do\s+a\s+)?(?:180|one eighty)",
                r"face\s+(?:the\s+other\s+way|behind)",
            ],
            "stop": [
                r"stop",
                r"halt",
                r"freeze",
                r"don't\s+move",
            ],
            # Look commands
            "look_at": [
                r"look\s+at\s+(?:the\s+)?(.+)",
                r"find\s+(?:the\s+)?(.+)",
                r"see\s+(?:the\s+)?(.+)",
                r"spot\s+(?:the\s+)?(.+)",
            ],
            "look_around": [
                r"look\s+around",
                r"scan\s+(?:the\s+)?(?:room|area|surroundings)",
                r"what\s+(?:do\s+you\s+see|can\s+you\s+see|is\s+around)",
            ],
            # Come commands
            "come_here": [
                r"come\s+(?:here|to\s+me)",
                r"follow\s+me",
                r"(?:come|move)\s+closer",
            ],
            # Emote commands
            "emote_happy": [
                r"(?:be|act)\s+(?:happy|excited|joyful)",
                r"(?:happy|excited|joyful)\s+(?:beep|sound)",
                r"celebrate",
                r"cheer(?:\s+up)?",
            ],
            "emote_sad": [
                r"(?:be|act)\s+(?:sad|unhappy)",
                r"(?:sad|unhappy)\s+(?:beep|sound)",
                r"(?:be|look)\s+disappointed",
            ],
            "emote_curious": [
                r"(?:be|act)\s+curious",
                r"curious\s+(?:beep|sound)",
                r"(?:be|act)\s+(?:interested|intrigued)",
            ],
            "emote_afraid": [
                r"(?:be|act)\s+(?:afraid|scared|frightened)",
                r"(?:afraid|scared)\s+(?:beep|sound)",
                r"(?:be|act)\s+(?:fearful|terrified)",
            ],
            "emote_hello": [
                r"(?:say|wave)\s+hello",
                r"(?:say|wave)\s+hi",
                r"greet\s+(?:me|us)",
                r"(?:hello|hi)",
            ],
            "emote_goodbye": [
                r"(?:say|wave)\s+(?:goodbye|bye)",
                r"farewell",
                r"(?:goodbye|bye)",
            ],
        }

        # Compile all regex patterns for efficiency
        self.compiled_patterns = {}
        for intent, patterns in self.patterns.items():
            self.compiled_patterns[intent] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        logger.debug(f"Initialized {len(self.patterns)} intent patterns")

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Parse text into a structured intent.

        Args:
            text: Natural language text to parse

        Returns:
            Dict with intent information or None if no intent matched
        """
        if not text or not text.strip():
            return None

        # Normalize text
        normalized_text = text.lower().strip()

        logger.debug(f"Parsing text: '{normalized_text}'")

        # Check for wake word
        wake_words = ["duck", "duckie", "ducky"]
        has_wake_word = any(word in normalized_text for word in wake_words)

        # Remove wake word from text for cleaner matching
        if has_wake_word:
            for word in wake_words:
                normalized_text = re.sub(r"\b" + word + r"\b", "", normalized_text)
            normalized_text = normalized_text.strip()
            logger.debug(f"Removed wake word, text: '{normalized_text}'")

        # Match against patterns
        for intent, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(normalized_text)
                if match:
                    result = self._build_intent_from_match(intent, match, normalized_text)
                    logger.info(f"Matched intent: {intent} -> {result}")
                    return result

        # No match found
        logger.debug(f"No intent match for: '{text}'")
        return None

    def _build_intent_from_match(self, intent: str, match: re.Match, text: str) -> Dict[str, Any]:
        """
        Build a structured intent from a regex match.

        Args:
            intent: Intent type
            match: Regex match object
            text: Original normalized text

        Returns:
            Dict with intent details
        """
        # Base structure
        result = {
            "intent_type": intent,
            "confidence": 0.8,  # Rule-based so use fixed confidence
            "original_text": text,
        }

        # Add action parameters based on intent type
        if intent.startswith("move_"):
            result["action_type"] = "move"
            result["params"] = {"direction": intent.replace("move_", "")}

        elif intent.startswith("turn_"):
            result["action_type"] = "turn"
            result["params"] = {"direction": intent.replace("turn_", "")}

        elif intent == "stop":
            result["action_type"] = "stop"
            result["params"] = {}

        elif intent == "look_at":
            result["action_type"] = "look_at"
            # Extract the target from the first regex group if available
            target = match.group(1) if match.groups() else "unknown"
            result["params"] = {"target": target}

        elif intent == "look_around":
            result["action_type"] = "look_around"
            result["params"] = {}

        elif intent == "come_here":
            result["action_type"] = "come_here"
            result["params"] = {}

        elif intent.startswith("emote_"):
            result["action_type"] = "emote"
            result["params"] = {"emote": intent.replace("emote_", "")}

        return result

    def get_available_intents(self) -> List[str]:
        """
        Get a list of available intent types.

        Returns:
            List of intent type strings
        """
        return list(self.patterns.keys())
