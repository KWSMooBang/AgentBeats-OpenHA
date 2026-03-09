"""
Session Manager for Purple Agent Server

Manages agent sessions and context state.
"""

import logging
from typing import Dict, Optional, Any
import time

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages sessions for A2A protocol.
    
    Attributes:
        sessions: Dict mapping session_id to session data
        contexts: Dict mapping context_id to context data
    """
    
    def __init__(self):
        """Initialize session manager."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.contexts: Dict[str, Dict[str, Any]] = {}
        logger.info("SessionManager initialized")
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session.
        
        Args:
            session_id: Unique session ID
        
        Returns:
            Session data dict
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": time.time(),
                "contexts": set(),
            }
            logger.info("Created session: %s", session_id)
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data.
        
        Args:
            session_id: Session ID to retrieve
        
        Returns:
            Session data dict or None if not found
        """
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session.
        
        Args:
            session_id: Session ID to delete
        """
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Delete all contexts in this session
            for context_id in session.get("contexts", set()):
                self.delete_context(context_id)
            del self.sessions[session_id]
            logger.info("Deleted session: %s", session_id)
    
    def create_context(self, context_id: str, session_id: str) -> Dict[str, Any]:
        """Create a new context.
        
        Args:
            context_id: Unique context ID
            session_id: Parent session ID
        
        Returns:
            Context data dict
        """
        if context_id not in self.contexts:
            self.contexts[context_id] = {
                "context_id": context_id,
                "session_id": session_id,
                "created_at": time.time(),
            }
            # Add context to session
            if session_id in self.sessions:
                self.sessions[session_id]["contexts"].add(context_id)
            logger.info("Created context: %s in session: %s", context_id, session_id)
        return self.contexts[context_id]
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """Get context data.
        
        Args:
            context_id: Context ID to retrieve
        
        Returns:
            Context data dict or None if not found
        """
        return self.contexts.get(context_id)
    
    def delete_context(self, context_id: str) -> None:
        """Delete a context.
        
        Args:
            context_id: Context ID to delete
        """
        if context_id in self.contexts:
            context = self.contexts[context_id]
            session_id = context.get("session_id")
            # Remove context from session
            if session_id and session_id in self.sessions:
                self.sessions[session_id]["contexts"].discard(context_id)
            del self.contexts[context_id]
            logger.info("Deleted context: %s", context_id)
    
    def list_sessions(self) -> list:
        """List all sessions.
        
        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())
    
    def list_contexts(self, session_id: Optional[str] = None) -> list:
        """List contexts, optionally filtered by session.
        
        Args:
            session_id: Optional session ID to filter by
        
        Returns:
            List of context IDs
        """
        if session_id:
            session = self.sessions.get(session_id)
            if session:
                return list(session.get("contexts", set()))
            return []
        return list(self.contexts.keys())
