"""
evaluation/evaluator.py

RAGAS-based evaluation framework for measuring agent output quality.

Measures:
- Answer faithfulness: Does the answer follow from the retrieved context?
- Answer relevancy: Does the answer address the question asked?
- Context precision: Is the retrieved context relevant to the question?
- Context recall: Does the retrieved context contain the answer?
"""

from __future__ import annotations 

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional 

from datasets import Dataset
from ragas import evaluate 
from ragas.metrics import (
    answer_faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from agent.agent import InsuranceAgent
from embeddings.store import VectorStore


@dataclass
class EvalSample:
    """A single evaluation sample — question, ground truth, and agent outputs."""
    question: str
    ground_truth: str
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    
@dataclass
class EvalResults:
    """Aggregated evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    n_samples: int
    
    def __str__(self) -> str:
        return (
            f"Evaluation Results (n={self.n_samples}):\n"
            f"{'-' * 40}\n"
            f"Faithfulness:       {self.faithfulness:.2f}\n"
            f"Answer Relevancy:   {self.answer_relevancy:.2f}\n"
            f"Context Precision:  {self.context_precision:.2f}\n"
            f"Context Recall:     {self.context_recall:.2f}\n"
        )
        
        
class AgentEvaluator:
    """
    Run RAGAS evaluation against the insurance agent.

    Usage:
        evaluator = AgentEvaluator()
        samples = evaluator.load_eval_set("data/processed/eval_set.json")
        results = evaluator.evaluate(samples)
        print(results)
    """
    
    def __init__(self):
        self._agent = InsuranceAgent(verrbose=False)
        self._store = VectorStore()
        
    def load_eval_set(self, path: str | Path) -> list[EvalSample]:
        """
        Load evaluation samples from a JSON file.

        Expected format:
        [
            {
                "question": "What is the deductible?",
                "ground_truth": "The deductible is $500 per year."
            },
            ...
        ]
        """
        data = json.loads(Path(path).read_text())
        return [EvalSample(q=d["question"], ground_truth=d["ground_truth"]) for d in data]
    
    def run_agent(self, samples: list[EvalSample]) -> list [EvalSample]:
        """Run the agent on each sample and collect answers and contexts."""
        print(f"Running agent on {len(samples)} samples...")
        for i, sample in enumerate(samples, 1):
            print(f". [{i}/{len(samples)}] {sample.question[:60]}...")
            
            # get agent answer
            result = self._agent.run_with_steps(sample.question)
            sample.answer = result["output"]
            
            # extract retrieved contexts from intermediate steps
            contexts = []
            for action, observation in result["intermediate_steps"]:
                if hasattr(action, "tool") and action.tool == "search_policy_document":
                    contexts.append(str(observation))
            sample.contexts = contexts if contexts else ["No context retrieved."]
            
            # reset memory between samples for clean evaluation
            self._agent.clear_memory()
            
        return samples

    def evalutate(self, samples: list[EvalSample]) -> EvalResults:
        """
        Run RAGAS evaluation metrics on a set of evaluated samples.

        Args:
            samples: Samples with answers and contexts already populated
                     (run run_agent() first).

        Returns:
            EvalResults with aggregated metric scores.
        """
        dataset = Dataset.from_dict({
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth for s in samples]
        })
        
        result = evaluate(
            dataset,
            metrics=[
                answer_faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ],
        )
        
        return EvalResults(
            faithfulness=result["faithfulness"],
            answer_relevancy=result["answer_relevancy"],
            context_precision=result["context_precision"],
            context_recall=result["context_recall"],
            n_samples=len(samples)
        )
