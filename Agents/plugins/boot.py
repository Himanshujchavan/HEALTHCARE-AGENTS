from .registry import register
from .diabetes import DiabetesPlugin
from .cardiovascular import CardiovascularPlugin

def init_plugins():
    register(DiabetesPlugin())
    register(CardiovascularPlugin())
