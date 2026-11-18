"""Secure external credential management for MicroDelay.

Guarantees that sensitive API keys, OAuth tokens, and client secrets are never
hardcoded, logged in plaintext, or exposed in API responses.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger("microdelay.integrations.credentials")


@dataclass
class OAuthToken:
    """Encapsulates an OAuth2 token with expiry tracking and refresh capability."""
    access_token: str
    token_type: str = "Bearer"
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    scope: Optional[str] = None
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self, buffer_seconds: int = 60) -> bool:
        """Check if token is expired or within buffer seconds of expiring."""
        if not self.expires_at:
            return False
        now = datetime.now(timezone.utc)
        target = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        return now >= (target - timedelta(seconds=buffer_seconds))


class CredentialManager:
    """Manages credentials across external services with redaction and environment fallback."""

    _instance: Optional["CredentialManager"] = None

    def __init__(self):
        # In-memory secure credential store: Dict[service_name, Dict[key_name, value]]
        self._store: Dict[str, Dict[str, str]] = {}
        # OAuth token store: Dict[service_name, Dict[user_or_client_id, OAuthToken]]
        self._tokens: Dict[str, Dict[str, OAuthToken]] = {}

    @classmethod
    def get_instance(cls) -> "CredentialManager":
        """Singleton access."""
        if cls._instance is None:
            cls._instance = CredentialManager()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (primarily for test isolation)."""
        cls._instance = CredentialManager()

    def set_credential(self, service: str, key_name: str, value: str) -> None:
        """Store a credential securely in-memory."""
        if not service or not key_name:
            raise ValueError("Service and key_name must be non-empty.")
        service_clean = service.strip().lower()
        key_clean = key_name.strip().upper()
        if service_clean not in self._store:
            self._store[service_clean] = {}
        self._store[service_clean][key_clean] = str(value).strip()
        logger.info(
            "Configured credential for service='%s', key='%s' (masked: %s)",
            service_clean, key_clean, self.mask_secret(value)
        )

    def get_credential(self, service: str, key_name: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a credential from in-memory store or environment variables.
        
        Lookup order:
        1. In-memory store: self._store[service][key_name]
        2. Environment variable: MICRODELAY_{SERVICE}_{KEY_NAME}
        3. Environment variable: {SERVICE}_{KEY_NAME}
        4. Environment variable: {KEY_NAME}
        5. Default value
        """
        service_clean = service.strip().lower()
        key_clean = key_name.strip().upper()

        # 1. Check in-memory store
        if service_clean in self._store and key_clean in self._store[service_clean]:
            return self._store[service_clean][key_clean]

        # 2. Check environment variables
        env_candidates = [
            f"MICRODELAY_{service_clean.upper()}_{key_clean}",
            f"{service_clean.upper()}_{key_clean}",
            key_clean
        ]
        for env_var in env_candidates:
            val = os.environ.get(env_var)
            if val is not None and val.strip() != "":
                return val.strip()

        return default

    def has_credential(self, service: str, key_name: str) -> bool:
        """Check if credential exists either in store or environment."""
        val = self.get_credential(service, key_name)
        return val is not None and len(val) > 0

    def delete_credential(self, service: str, key_name: str) -> bool:
        """Remove a credential from the in-memory store."""
        service_clean = service.strip().lower()
        key_clean = key_name.strip().upper()
        if service_clean in self._store and key_clean in self._store[service_clean]:
            del self._store[service_clean][key_clean]
            return True
        return False

    def set_token(self, service: str, subject_id: str, token: OAuthToken) -> None:
        """Store an OAuth token."""
        service_clean = service.strip().lower()
        if service_clean not in self._tokens:
            self._tokens[service_clean] = {}
        self._tokens[service_clean][subject_id] = token

    def get_token(self, service: str, subject_id: str) -> Optional[OAuthToken]:
        """Retrieve an OAuth token."""
        service_clean = service.strip().lower()
        return self._tokens.get(service_clean, {}).get(subject_id)

    @staticmethod
    def mask_secret(secret: Optional[str]) -> str:
        """Redact a secret string showing only first and last few characters."""
        if not secret:
            return "<none>"
        secret_str = str(secret).strip()
        length = len(secret_str)
        if length <= 6:
            return "***"
        if length <= 12:
            return f"{secret_str[:2]}...{secret_str[-2:]}"
        return f"{secret_str[:4]}...{secret_str[-4:]}"

    def get_masked_inventory(self) -> Dict[str, Dict[str, str]]:
        """Return inventory of configured credentials with masked values."""
        inventory: Dict[str, Dict[str, str]] = {}
        for s_name, keys in self._store.items():
            inventory[s_name] = {k: self.mask_secret(v) for k, v in keys.items()}
        return inventory
