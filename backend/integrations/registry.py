"""Integration Registry and Lifecycle Manager for MicroDelay.

Enables independent registration, activation, deactivation, and health monitoring
for all external service integrations (Calendar, Travel, Traffic, Weather).
"""

import logging
from typing import Dict, Any, Optional, List, Type
from datetime import datetime, timezone

from .base import BaseIntegration, IntegrationStatus, IntegrationHealth

logger = logging.getLogger("microdelay.integrations.registry")


class IntegrationRegistry:
    """Central registry managing the lifecycle of all external integrations."""

    _instance: Optional["IntegrationRegistry"] = None

    def __init__(self):
        # Maps category -> dict of provider_name -> BaseIntegration
        # Categories: "calendar", "travel", "traffic", "weather", "notification"
        self._integrations: Dict[str, Dict[str, BaseIntegration]] = {
            "calendar": {},
            "travel": {},
            "traffic": {},
            "weather": {},
            "notification": {}
        }
        # Active default provider per category
        self._default_providers: Dict[str, Optional[str]] = {
            "calendar": None,
            "travel": None,
            "traffic": None,
            "weather": None,
            "notification": None
        }

    @classmethod
    def get_instance(cls) -> "IntegrationRegistry":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = IntegrationRegistry()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton for test isolation."""
        cls._instance = IntegrationRegistry()

    def register(
        self,
        category: str,
        integration: BaseIntegration,
        is_default: bool = False
    ) -> None:
        """Register an integration adapter under a category."""
        category = category.lower().strip()
        if category not in self._integrations:
            self._integrations[category] = {}

        provider = integration.provider_name.lower().strip()
        self._integrations[category][provider] = integration

        if is_default or self._default_providers.get(category) is None:
            self._default_providers[category] = provider

        logger.info(
            "Registered %s integration: provider='%s' (default=%s)",
            category, provider, self._default_providers.get(category) == provider
        )

    def get(
        self,
        category: str,
        provider_name: Optional[str] = None
    ) -> Optional[BaseIntegration]:
        """Retrieve an integration by category and provider name."""
        category = category.lower().strip()
        cat_dict = self._integrations.get(category, {})

        if not cat_dict:
            return None

        if provider_name:
            return cat_dict.get(provider_name.lower().strip())

        # Return default if configured
        default_name = self._default_providers.get(category)
        if default_name and default_name in cat_dict:
            return cat_dict[default_name]

        # Return first available
        if cat_dict:
            return next(iter(cat_dict.values()))

        return None

    def set_default(self, category: str, provider_name: str) -> bool:
        """Set the active default provider for a category."""
        category = category.lower().strip()
        provider_name = provider_name.lower().strip()
        if category in self._integrations and provider_name in self._integrations[category]:
            self._default_providers[category] = provider_name
            return True
        return False

    def enable_integration(self, category: str, provider_name: Optional[str] = None) -> bool:
        """Enable a specific integration or all integrations in a category."""
        integration = self.get(category, provider_name)
        if integration:
            integration.enable()
            return True
        return False

    def disable_integration(self, category: str, provider_name: Optional[str] = None) -> bool:
        """Disable a specific integration or all integrations in a category."""
        integration = self.get(category, provider_name)
        if integration:
            integration.disable()
            return True
        return False

    def is_category_available(self, category: str) -> bool:
        """Check if at least one healthy/enabled provider exists for category."""
        category = category.lower().strip()
        cat_dict = self._integrations.get(category, {})
        for integration in cat_dict.values():
            if integration.is_enabled():
                return True
        return False

    def get_all_health(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Collect health status across all registered integrations."""
        report: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for category, providers in self._integrations.items():
            report[category] = {}
            for provider_name, integration in providers.items():
                health = integration.health_check()
                report[category][provider_name] = {
                    "status": health.status.value,
                    "enabled": integration.is_enabled(),
                    "message": health.message,
                    "is_default": (self._default_providers.get(category) == provider_name),
                    "checked_at": health.checked_at,
                    "details": health.details
                }
        return report
