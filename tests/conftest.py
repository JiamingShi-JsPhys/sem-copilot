"""Test-suite root for the standalone sem-copilot repo.

Puts the repo root on ``sys.path`` so ``from lib.chelate_core import ...``
and ``import contracts`` resolve when a test loads a skill script by
absolute path.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = REPO_ROOT / "skills"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
