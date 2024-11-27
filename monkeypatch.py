from typing import Any
from mteb.evaluation.evaluators import RetrievalEvaluator
from vidore_benchmark.retrievers.vision_retriever import VisionRetriever
"""
In your main script or wherever you're using the benchmark, just import this at the top:
import monkeypatch
"""

def compute_metrics_fixed(self, relevant_docs: Any, results: Any, **kwargs):
    """Fixed compute_metrics method that properly initializes the evaluator"""
    mteb_evaluator = RetrievalEvaluator(retriever=self)

    ndcg, _map, recall, precision, naucs = mteb_evaluator.evaluate(
        relevant_docs,
        results,
        mteb_evaluator.k_values,
        ignore_identical_ids=kwargs.get("ignore_identical_ids", True),
    )

    mrr = mteb_evaluator.evaluate_custom(relevant_docs, results, mteb_evaluator.k_values, "mrr")

    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
        **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr[0].items()},
        **{f"naucs_at_{k.split('@')[1]}": v for (k, v) in naucs.items()},
    }

    return scores

# Apply the monkeypatch
VisionRetriever.compute_metrics = compute_metrics_fixed
