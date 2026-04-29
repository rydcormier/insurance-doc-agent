"""
scripts/evaluate.py

Run RAGAS on the agent.

Usage:
    python scripts/evaluate.py --dataset data/processed/eval_set.json
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.getLogger("ragas").setLevel(logging.ERROR)

from evaluation.evaluator import AgentEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate agent output quality.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset (JSON format)."
    )
    args = parser.parse_args()
    
    evaluator = AgentEvaluator()
    samples = evaluator.load_eval_set(args.dataset)
    samples = evaluator.run_agent(samples)
    results = evaluator.evaluate(samples)
    print(results)
    

if __name__ == "__main__":
    main()
