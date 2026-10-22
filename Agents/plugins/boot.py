"""Plugin bootstrap.

Registers the built-in disease plugins exactly once.
"""

from .registry import register
from .diabetes import DiabetesPlugin
from .cardiovascular import CardiovascularPlugin

_initialized = False


def init_plugins():
    """Register the built-in disease plugins (idempotent)."""
    global _initialized
    if _initialized:
        return
    _initialized = True
    register(DiabetesPlugin())
    register(CardiovascularPlugin())
