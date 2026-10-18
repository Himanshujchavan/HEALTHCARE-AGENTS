from typing import List, Dict, Any, Optional
from .base import DiseasePlugin

# Global registry for disease plugins
_plugins: Dict[str, DiseasePlugin] = {}

def register(plugin: DiseasePlugin):
    """Registers a disease plugin by its name."""
    _plugins[plugin.name] = plugin

def get(name: str) -> Optional[DiseasePlugin]:
    """Retrieves a registered plugin by name."""
    return _plugins.get(name)

def applicable(health_data: Dict[str, Any]) -> List[DiseasePlugin]:
    """
    Returns all plugins whose required fields are present in the health data and not None.
    """
    applicable_plugins = []
    for plugin in _plugins.values():
        if all(health_data.get(field) is not None for field in plugin.required_fields):
            applicable_plugins.append(plugin)
    return applicable_plugins

def list_all() -> List[str]:
    """Returns names of all registered plugins."""
    return list(_plugins.keys())
