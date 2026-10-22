"""Disease plugin package.

Importing this package registers the built-in plugins, so
``from Agents.plugins import registry`` always sees them.
"""

from .base import DiseasePlugin
from .registry import applicable, get, list_all, register

__all__ = [
    "DiseasePlugin",
    "applicable",
    "get",
    "list_all",
    "register",
]
