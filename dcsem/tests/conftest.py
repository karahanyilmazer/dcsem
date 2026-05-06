"""Pytest configuration for dcsem tests."""

import sys
from pathlib import Path

# Add the project root to the Python path so dcsem can be imported
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
