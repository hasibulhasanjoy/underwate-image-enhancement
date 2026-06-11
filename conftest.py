"""pytest configuration — makes the project root importable as a package root.

Place this file at ~/underwater_image_enhancement/conftest.py (the project
root, NOT inside tests/ or src/).

pytest loads conftest.py before any test collection, so inserting the project
root onto sys.path here ensures that `import src.models`, `import src.physics`,
etc. all resolve correctly regardless of how pytest is invoked.
"""

import sys
from pathlib import Path

# Insert project root at the front of sys.path so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).parent))
