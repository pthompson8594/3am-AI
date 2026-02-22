#!/usr/bin/env python3
"""
Authentication - Multi-user authentication system.

Provides:
- User registration with password hashing
- Login with session tokens
- Session management
- Per-user data isolation
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import re

from data_security import SecureUserData, generate_salt


DATA_DIR = Path.home() / ".local/share/llm-unified"
USERS_FILE = DATA_DIR / "users.json"
SESSIONS_FILE = DATA_DIR / "sessions.json"


@dataclass
class User:
    """User account."""
    id: str
    username: str
    password_hash: str
    salt: str
    created_at: float
    last_login: float = 0
    settings: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "username": self.username,
            "password_hash": self.password_hash,
            "salt": self.salt,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "settings": self.settings,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in cls.__dataclass_fields__})
    
    def to_public_dict(self) -> dict:
        """Return user data safe to expose to client."""
        return {
            "id": self.id,
            "username": self.username,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "settings": self.settings,
        }


@dataclass
class Session:
    """User session."""
    token: str
    user_id: str
    created_at: float
    expires_at: float
    last_activity: float
    
    def to_dict(self) -> dict:
        return {
            "token": self.token,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_activity": self.last_activity,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(**data)
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class AuthError(Exception):
    """Authentication error."""
    pass


class AuthSystem:
    """
    Multi-user authentication system.
    
    - Password hashing with salt (bcrypt planned, SHA-256 for now)
    - Session token management
    - Automatic session cleanup
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        self.session_timeout_hours = session_timeout_hours
        self.users: dict[str, User] = {}  # username -> User
        self.sessions: dict[str, Session] = {}  # token -> Session
        
        self._load()
    
    def _load(self):
        """Load users and sessions from disk."""
        try:
            if USERS_FILE.exists():
                with open(USERS_FILE) as f:
                    data = json.load(f)
                self.users = {
                    u["username"]: User.from_dict(u)
                    for u in data.get("users", [])
                }
        except Exception as e:
            print(f"[Auth] Error loading users: {e}")
        
        try:
            if SESSIONS_FILE.exists():
                with open(SESSIONS_FILE) as f:
                    data = json.load(f)
                self.sessions = {
                    s["token"]: Session.from_dict(s)
                    for s in data.get("sessions", [])
                    if not Session.from_dict(s).is_expired()
                }
        except Exception as e:
            print(f"[Auth] Error loading sessions: {e}")
    
    def _save_users(self):
        """Save users to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {"users": [u.to_dict() for u in self.users.values()]}
            with open(USERS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Auth] Error saving users: {e}")
    
    def _save_sessions(self):
        """Save sessions to disk."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Only save non-expired sessions
            valid_sessions = [s.to_dict() for s in self.sessions.values() if not s.is_expired()]
            data = {"sessions": valid_sessions}
            with open(SESSIONS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Auth] Error saving sessions: {e}")
    
    def _hash_password(self, password: str, salt: str) -> str:
        """
        Hash a password with salt.
        
        TODO: Replace with bcrypt for production:
            import bcrypt
            return bcrypt.hashpw(password.encode(), salt.encode()).decode()
        """
        # SHA-256 placeholder (use bcrypt in production)
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def _generate_user_id(self) -> str:
        """Generate a unique user ID."""
        return secrets.token_hex(16)
    
    def _generate_session_token(self) -> str:
        """Generate a secure session token."""
        return secrets.token_urlsafe(32)
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format."""
        if not username or len(username) < 3 or len(username) > 32:
            return False
        # Alphanumeric, underscores, hyphens only
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', username))
    
    def _validate_password(self, password: str) -> tuple[bool, str]:
        """Validate password strength."""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        if len(password) > 128:
            return False, "Password too long"
        # TODO: Add more password requirements
        return True, ""
    
    def register(self, username: str, password: str) -> User:
        """
        Register a new user.
        
        Args:
            username: Desired username (3-32 chars, alphanumeric)
            password: Password (min 8 chars)
            
        Returns:
            Created User object
            
        Raises:
            AuthError: If registration fails
        """
        # Validate username
        if not self._validate_username(username):
            raise AuthError("Invalid username. Use 3-32 alphanumeric characters, underscores, or hyphens.")
        
        # Check if username exists
        if username.lower() in [u.lower() for u in self.users.keys()]:
            raise AuthError("Username already exists")
        
        # Validate password
        valid, msg = self._validate_password(password)
        if not valid:
            raise AuthError(msg)
        
        # Create user
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)
        
        user = User(
            id=self._generate_user_id(),
            username=username,
            password_hash=password_hash,
            salt=salt,
            created_at=time.time(),
        )
        
        self.users[username] = user
        self._save_users()
        
        # Create user data directory
        user_dir = DATA_DIR / "users" / user.id
        user_dir.mkdir(parents=True, exist_ok=True)
        
        return user
    
    def login(self, username: str, password: str) -> tuple[User, str]:
        """
        Login a user.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple of (User, session_token)
            
        Raises:
            AuthError: If login fails
        """
        user = self.users.get(username)
        if not user:
            raise AuthError("Invalid username or password")
        
        # Verify password
        password_hash = self._hash_password(password, user.salt)
        if not hmac.compare_digest(password_hash, user.password_hash):
            raise AuthError("Invalid username or password")
        
        # Update last login
        user.last_login = time.time()
        self._save_users()
        
        # Create session
        token = self._generate_session_token()
        session = Session(
            token=token,
            user_id=user.id,
            created_at=time.time(),
            expires_at=time.time() + (self.session_timeout_hours * 3600),
            last_activity=time.time(),
        )
        
        self.sessions[token] = session
        self._save_sessions()
        
        return user, token
    
    def logout(self, token: str) -> bool:
        """
        Logout a user by invalidating their session.
        
        Args:
            token: Session token
            
        Returns:
            True if session was found and removed
        """
        if token in self.sessions:
            del self.sessions[token]
            self._save_sessions()
            return True
        return False
    
    def validate_session(self, token: str) -> Optional[User]:
        """
        Validate a session token and return the user.
        
        Args:
            token: Session token
            
        Returns:
            User if session is valid, None otherwise
        """
        session = self.sessions.get(token)
        if not session:
            return None
        
        if session.is_expired():
            del self.sessions[token]
            self._save_sessions()
            return None
        
        # Update last activity
        session.last_activity = time.time()
        
        # Find user
        for user in self.users.values():
            if user.id == session.user_id:
                return user
        
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by their ID."""
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None
    
    def get_user_data(self, user: User) -> SecureUserData:
        """Get the secure data handler for a user."""
        return SecureUserData(user.id)
    
    def update_user_settings(self, user: User, settings: dict):
        """Update user settings."""
        user.settings.update(settings)
        self._save_users()
    
    def change_password(self, user: User, old_password: str, new_password: str) -> bool:
        """
        Change a user's password.
        
        Args:
            user: User object
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password was changed
            
        Raises:
            AuthError: If old password is wrong or new password invalid
        """
        # Verify old password
        old_hash = self._hash_password(old_password, user.salt)
        if not hmac.compare_digest(old_hash, user.password_hash):
            raise AuthError("Current password is incorrect")
        
        # Validate new password
        valid, msg = self._validate_password(new_password)
        if not valid:
            raise AuthError(msg)
        
        # Update password
        new_salt = secrets.token_hex(16)
        user.salt = new_salt
        user.password_hash = self._hash_password(new_password, new_salt)
        self._save_users()
        
        # Invalidate all sessions for this user
        to_remove = [t for t, s in self.sessions.items() if s.user_id == user.id]
        for t in to_remove:
            del self.sessions[t]
        self._save_sessions()
        
        return True
    
    def delete_user(self, user: User, password: str) -> bool:
        """
        Delete a user account.
        
        Args:
            user: User to delete
            password: Password for confirmation
            
        Returns:
            True if user was deleted
            
        Raises:
            AuthError: If password is wrong
        """
        # Verify password
        password_hash = self._hash_password(password, user.salt)
        if not hmac.compare_digest(password_hash, user.password_hash):
            raise AuthError("Password is incorrect")
        
        # Remove sessions
        to_remove = [t for t, s in self.sessions.items() if s.user_id == user.id]
        for t in to_remove:
            del self.sessions[t]
        
        # Remove user
        del self.users[user.username]
        
        self._save_users()
        self._save_sessions()
        
        # Note: User data directory is NOT deleted automatically
        # Could add optional data deletion here
        
        return True
    
    def cleanup_expired_sessions(self):
        """Remove all expired sessions."""
        expired = [t for t, s in self.sessions.items() if s.is_expired()]
        for t in expired:
            del self.sessions[t]
        if expired:
            self._save_sessions()
        return len(expired)
    
    def get_all_users(self) -> list[dict]:
        """Get list of all users (public info only)."""
        return [u.to_public_dict() for u in self.users.values()]
    
    def user_count(self) -> int:
        """Get total number of registered users."""
        return len(self.users)
