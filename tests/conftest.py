"""Pytest configuration for AI-HEALTH project.

Ensures the project root (containing the Agents/, app/, etc. folders)
 is on sys.path so imports like `from Agents...` work in tests.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
