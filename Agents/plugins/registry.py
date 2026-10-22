"""Disease plugin registry.

The registry is the bridge between the Master Health Agent and the
disease-specific plugins. It is self-initializing: importing the registry
is enough to make the built-in plugins available, no matter which module
was imported first.
"""

from typing import List, Dict, Any, Optional
from .base import DiseasePlugin

# Global registry for disease plugins
_plugins: Dict[str, DiseasePlugin] = {}
_initialized = False


def _ensure_initialized() -> None:
    """Register the built-in plugins once, on first use."""
    global _initialized
    if _initialized:
        return
    _initialized = True
    # Imported lazily to avoid a circular import: boot imports the plugins,
    # the plugins import this registry for lookups.
    from .boot import init_plugins
    init_plugins()


def register(plugin: DiseasePlugin):
    """Registers a disease plugin by its name."""
    _plugins[plugin.name] = plugin
    return plugin


def get(name: str) -> Optional[DiseasePlugin]:
    """Retrieves a registered plugin by name."""
    _ensure_initialized()
    return _plugins.get(name)


def applicable(health_data: Dict[str, Any]) -> List[DiseasePlugin]:
    """
    Returns all plugins whose required fields are present in the health data and not None.
    """
    _ensure_initialized()
    applicable_plugins = []
    for plugin in _plugins.values():
        if all(health_data.get(field) is not None for field in plugin.required_fields):
            applicable_plugins.append(plugin)
    return applicable_plugins


def list_all() -> List[str]:
    """Returns names of all registered plugins."""
    _ensure_initialized()
    return list(_plugins.keys())


def missing_fields(health_data: Dict[str, Any], plugin: DiseasePlugin) -> List[str]:
    """Returns the required fields of ``plugin`` that are absent from ``health_data``."""
    return [
        field for field in plugin.required_fields
        if health_data.get(field) is None
    ]
