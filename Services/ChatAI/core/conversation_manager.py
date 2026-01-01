"""
Conversation Manager for Financial Advisor Chatbot

Maintains session memory for multi-turn conversations:
- Tracks conversation history
- Remembers active client being discussed
- Provides context window for follow-up questions
"""

import re
from datetime import datetime, timedelta
from typing import Optional


class ConversationManager:
    """Manages conversation state for a user session."""
    
    def __init__(self, session_id: str, ttl_minutes: int = 30):
        self.session_id = session_id
        self.history: list[dict] = []
        self.active_client: Optional[str] = None
        self.active_scheme: Optional[str] = None
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now() - self.last_activity > self.ttl
    
    def touch(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        intent: Optional[str] = None
    ):
        """
        Add a message to conversation history.
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            intent: Classified intent type (optional)
        """
        self.history.append({
            "role": role,
            "content": content,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        })
        self.touch()
        
        # Extract and remember entities
        if client := self._extract_client_name(content):
            self.active_client = client
        if scheme := self._extract_scheme_name(content):
            self.active_scheme = scheme
    
    def get_context_window(self, n: int = 5) -> str:
        """
        Get recent conversation history as context string.
        
        Args:
            n: Number of recent messages to include
        
        Returns:
            Formatted string of recent conversation
        """
        recent = self.history[-n:]
        return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in recent])
    
    def get_active_context(self) -> dict:
        """Get currently active entities."""
        return {
            "client": self.active_client,
            "scheme": self.active_scheme
        }
    
    def _extract_client_name(self, text: str) -> Optional[str]:
        """
        Extract client name from text.
        
        Patterns:
        - "Sharma ji", "Patel ji"
        - "Mr. Sharma", "Mrs. Gupta"
        - "client Agarwal", "for Mehta's portfolio"
        """
        patterns = [
            r"(?:client|mr\.|mrs\.|ms\.)\s+(\w+)",
            r"(\w+)\s+(?:ji|sir|madam)",
            r"for\s+(\w+)(?:'s)?\s+(?:portfolio|holdings|sip)",
            r"(\w+)(?:'s)?\s+(?:investment|account)"
        ]
        for pattern in patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                name = match.group(1)
                # Filter out common words that aren't names
                if name.lower() not in ['the', 'a', 'an', 'my', 'our', 'their']:
                    return name.capitalize()
        return None
    
    def _extract_scheme_name(self, text: str) -> Optional[str]:
        """Extract mutual fund scheme name from text."""
        # Common MF house names
        mf_houses = r"(?:sbi|hdfc|icici|axis|kotak|mirae|nippon|dsp|tata|aditya birla|franklin|uti)"
        patterns = [
            rf"({mf_houses}\s+\w+(?:\s+\w+)?(?:\s+fund)?)",
            r"((?:bluechip|flexi cap|large cap|mid cap|small cap|elss)\s+fund)"
        ]
        for pattern in patterns:
            if match := re.search(pattern, text, re.IGNORECASE):
                return match.group(1).title()
        return None
    
    def clear(self):
        """Clear conversation history."""
        self.history = []
        self.active_client = None
        self.active_scheme = None


# In-memory session store (use Redis in production)
_sessions: dict[str, ConversationManager] = {}


def get_or_create_session(session_id: str) -> ConversationManager:
    """
    Get existing session or create new one.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        ConversationManager instance
    """
    if session_id not in _sessions or _sessions[session_id].is_expired():
        _sessions[session_id] = ConversationManager(session_id)
    else:
        _sessions[session_id].touch()
    return _sessions[session_id]


def cleanup_expired_sessions():
    """Remove expired sessions from memory."""
    expired = [
        sid for sid, session in _sessions.items() 
        if session.is_expired()
    ]
    for sid in expired:
        del _sessions[sid]
