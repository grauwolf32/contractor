"""Reject ADK feature modes outside the supported Runtime integration."""

from __future__ import annotations

from google.adk import features

_UNSUPPORTED_FEATURES = (
    features.FeatureName.FUNCTION_TOOL_ARG_VALIDATION,
    features.FeatureName.IN_MEMORY_SESSION_SERVICE_LIGHT_COPY,
    features.FeatureName.SKILL_LIFECYCLE,
)


def require_supported_adk_features() -> None:
    """Fail before resources are created when ADK enables an unsupported mode."""

    for feature in _UNSUPPORTED_FEATURES:
        # The public getter covers defaults, environment and programmatic overrides.
        # With warning-strict startup an experimental warning raises before the
        # getter returns; reject that path too, without suppressing it to run on.
        try:
            enabled = features.is_feature_enabled(feature)
        except UserWarning:
            raise RuntimeError(f"Unsupported ADK feature enabled: {feature.value}") from None
        if enabled:
            raise RuntimeError(f"Unsupported ADK feature enabled: {feature.value}")
