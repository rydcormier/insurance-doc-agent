"""
scripts/evaluate.py

Run RAGAS on the agent.

Usage:
    python scripts/evaluate.py --dataset data/processed/eval_set.json
"""

import argparse
import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

