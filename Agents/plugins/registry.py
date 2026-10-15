from typing import List, Dict, Any, Optional
from Agents.plugins.base import DiseasePlugin

_plugins: List[DiseasePlugin] = []

def register(plugin: DiseasePlugin):
    """Register a disease plugin into the global registry."""
    if plugin not in _plugins:
        _plugins.append(plugin)

def get(name: str) -> Optional[DiseasePlugin]:
    """Retrieve a plugin by its name."""
    for plugin in _plugins:
        if plugin.name == name:
            return plugin
    return None

def all() -> List[DiseasePlugin]:
    """Return all registered plugins in order of registration."""
    return _plugins

def applicable(health_data: Dict[str, Any]) -> List[DiseasePlugin]:
    """
    Return plugins whose required_fields are all present in the provided health data.
    Returns plugins in registration order.
    """
    applicable_plugins = []
    for plugin in _plugins:
        if all(field in health_data for field in plugin.required_fields):
            applicable_plugins.append(plugin)
    return applicable_plugins
