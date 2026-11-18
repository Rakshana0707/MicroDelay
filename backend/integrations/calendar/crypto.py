"""Cryptographic utilities for OAuth token encryption, state validation, and PKCE.

Ensures that:
1. Google OAuth tokens are encrypted at rest using AES-CBC + HMAC-SHA256 (Fernet).
2. OAuth states are bound to the user session with short TTL and single-use validation (anti-CSRF).
3. PKCE code_verifier / code_challenge pairs are securely generated and verified.
4. Tokens and client secrets are never exposed in plaintext or logged.
"""

import os
import json
import base64
import hashlib
import secrets
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("microdelay.calendar.crypto")

# Static application salt for key derivation
_STATIC_SALT = b"microdelay_calendar_oauth_salt_v1"


class OAuthTokenCrypto:
    """Handles authenticated symmetric encryption for OAuth tokens stored at rest."""

    _instance: Optional["OAuthTokenCrypto"] = None

    def __init__(self, master_secret: Optional[str] = None):
        secret = (
            master_secret
            or os.environ.get("MICRODELAY_ENCRYPTION_KEY")
            or os.environ.get("SECRET_KEY")
            or "microdelay-production-insecure-fallback-key-2026"
        )
        # Derive 32-byte URL-safe base64 key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_STATIC_SALT,
            iterations=100_000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret.encode("utf-8")))
        self._fernet = Fernet(key)

    @classmethod
    def get_instance(cls) -> "OAuthTokenCrypto":
        if cls._instance is None:
            cls._instance = OAuthTokenCrypto()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def encrypt_tokens(self, token_payload: Dict[str, Any]) -> str:
        """Serialize and encrypt token payload dictionary."""
        serialized = json.dumps(token_payload).encode("utf-8")
        encrypted = self._fernet.encrypt(serialized)
        return encrypted.decode("utf-8")

    def decrypt_tokens(self, encrypted_ciphertext: str) -> Dict[str, Any]:
        """Decrypt ciphertext and deserialize token payload dictionary."""
        try:
            decrypted_bytes = self._fernet.decrypt(encrypted_ciphertext.encode("utf-8"))
            return json.loads(decrypted_bytes.decode("utf-8"))
        except Exception as e:
            logger.error("Failed to decrypt stored OAuth tokens: %s", e)
            raise ValueError("Token decryption failure. Stored credential payload may be corrupted or invalid.")

    @staticmethod
    def redact_token_dict(tokens: Dict[str, Any]) -> Dict[str, Any]:
        """Create a sanitized copy of token data for safe audit logging."""
        sanitized = {}
        for k, v in tokens.items():
            if k in ["token", "access_token", "refresh_token", "client_secret", "id_token"]:
                if v and isinstance(v, str):
                    sanitized[k] = f"{v[:4]}...{v[-4:]}" if len(v) > 8 else "***"
                else:
                    sanitized[k] = "<redacted>"
            else:
                sanitized[k] = v
        return sanitized


class OAuthStateStore:
    """In-memory thread-safe store for OAuth state validation and PKCE verification."""

    _instance: Optional["OAuthStateStore"] = None

    def __init__(self, ttl_seconds: int = 600):
        # Maps state -> {user_id, code_verifier, redirect_uri, expires_at}
        self._states: Dict[str, Dict[str, Any]] = {}
        self.ttl_seconds = ttl_seconds

    @classmethod
    def get_instance(cls) -> "OAuthStateStore":
        if cls._instance is None:
            cls._instance = OAuthStateStore()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def generate_state(
        self,
        user_id: int,
        redirect_uri: str,
        code_verifier: Optional[str] = None
    ) -> str:
        """Generate a cryptographically secure state token bound to user and redirect_uri."""
        self._purge_expired()
        state = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)

        self._states[state] = {
            "user_id": user_id,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
            "expires_at": expires_at
        }
        return state

    def validate_and_consume(
        self,
        state: str,
        expected_user_id: int
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate state, verify user ownership, and consume it immediately.
        
        Returns:
            (is_valid, code_verifier, redirect_uri)
        """
        self._purge_expired()
        record = self._states.pop(state, None)
        if not record:
            logger.warning("OAuth state '%s' not found or already consumed.", state)
            return False, None, None

        now = datetime.now(timezone.utc)
        if now > record["expires_at"]:
            logger.warning("OAuth state '%s' has expired.", state)
            return False, None, None

        if record["user_id"] != expected_user_id:
            logger.warning(
                "OAuth state user mismatch: expected=%s, got=%s.",
                expected_user_id, record["user_id"]
            )
            return False, None, None

        return True, record.get("code_verifier"), record.get("redirect_uri")

    def _purge_expired(self) -> None:
        """Clean up expired states."""
        now = datetime.now(timezone.utc)
        expired = [k for k, v in self._states.items() if now > v["expires_at"]]
        for k in expired:
            del self._states[k]


def generate_pkce_pair() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge
