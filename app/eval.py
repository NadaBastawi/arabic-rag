"""
Evaluation script for Arabic RAG system.

Provides metrics: Recall@K, MRR, NDCG for retrieval evaluation.
"""

import json
from typing import List, Dict, Tuple
import numpy as np
import logging

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import SimpleVectorStore
from app.services.rag_pipeline import RAGPipeline
from app.services.multi_stage_retrieval import ArabicReRanker, MultiStageRetriever

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluator for Arabic RAG retrieval quality."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: SimpleVectorStore
    ):
        """Initialize evaluator."""
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    def recall_at_k(
        self,
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_ids: List of retrieved document IDs.
            relevant_ids: List of relevant document IDs.
            k: Cutoff position.
            
        Returns:
            Recall@K score (0 to 1).
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        intersection = retrieved_k & relevant_set
        return len(intersection) / len(relevant_set)
    
    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[int],
        relevant_ids: List[int]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank).
        
        Args:
            retrieved_ids: List of retrieved document IDs.
            relevant_ids: List of relevant document IDs.
            
        Returns:
            MRR score (0 to 1).
        """
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    def ndcg_at_k(
        self,
        retrieved_ids: List[int],
        relevant_ids: List[int],
        k: int
    ) -> float:
        """
        Calculate NDCG@K (Normalized Discounted Cumulative Gain).
        
        Args:
            retrieved_ids: List of retrieved document IDs.
            relevant_ids: List of relevant document IDs.
            k: Cutoff position.
            
        Returns:
            NDCG@K score (0 to 1).
        """
        if not relevant_ids:
            return 0.0
        
        relevant_set = set(relevant_ids)
        
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)
        
        idcg = 0.0
        for rank in range(1, min(k, len(relevant_ids)) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg == 0.0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate(
        self,
        queries: List[str],
        relevant_docs: List[List[int]],
        retrieval_mode: str = "basic",
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate retrieval on a set of queries.
        
        Args:
            queries: List of query strings.
            relevant_docs: List of relevant document ID lists (one per query).
            retrieval_mode: Retrieval mode to evaluate.
            top_k: K value for metrics.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if len(queries) != len(relevant_docs):
            raise ValueError("Number of queries must match number of relevant docs")
        
        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        
        for query, rel_ids in zip(queries, relevant_docs):
            try:
                if retrieval_mode == "multi_stage":
                    reranker = ArabicReRanker(**config.RE_RANKER_WEIGHTS)
                    retriever = MultiStageRetriever(
                        self.embedding_service,
                        self.vector_store,
                        reranker=reranker
                    )
                    results = retriever.retrieve(query, top_k=top_k)
                else:
                    q_emb = self.embedding_service.embed(query)
                    results = self.vector_store.search(q_emb, top_k=top_k)
                
                retrieved_ids = list(range(len(results)))
                
                recall = self.recall_at_k(retrieved_ids, rel_ids, top_k)
                mrr = self.mean_reciprocal_rank(retrieved_ids, rel_ids)
                ndcg = self.ndcg_at_k(retrieved_ids, rel_ids, top_k)
                
                recall_scores.append(recall)
                mrr_scores.append(mrr)
                ndcg_scores.append(ndcg)
                
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {str(e)}")
                continue
        
        return {
            f"recall@{top_k}": np.mean(recall_scores) if recall_scores else 0.0,
            "mrr": np.mean(mrr_scores) if mrr_scores else 0.0,
            f"ndcg@{top_k}": np.mean(ndcg_scores) if ndcg_scores else 0.0,
            "num_evaluated": len(recall_scores)
        }
    
    def compare_modes(
        self,
        queries: List[str],
        relevant_docs: List[List[int]],
        top_k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare basic vs multi_stage retrieval.
        
        Args:
            queries: List of query strings.
            relevant_docs: List of relevant document ID lists.
            top_k: K value for metrics.
            
        Returns:
            Dictionary with metrics for each mode.
        """
        logger.info("Evaluating basic retrieval...")
        basic_metrics = self.evaluate(
            queries, relevant_docs, "basic", top_k
        )
        
        logger.info("Evaluating multi_stage retrieval...")
        multi_stage_metrics = self.evaluate(
            queries, relevant_docs, "multi_stage", top_k
        )
        
        return {
            "basic": basic_metrics,
            "multi_stage": multi_stage_metrics
        }


def load_test_data(path: str) -> Tuple[List[str], List[List[int]]]:
    """
    Load test data from JSON file.
    
    Args:
        path: Path to test data JSON file.
        
    Returns:
        Tuple of (queries, relevant_docs).
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = [item['query'] for item in data]
    relevant_docs = [item['relevant_docs'] for item in data]
    
    return queries, relevant_docs


def main():
    """Run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Arabic RAG")
    parser.add_argument("--test-data", type=str, help="Path to test data JSON")
    parser.add_argument("--top-k", type=int, default=5, help="K value for metrics")
    parser.add_argument("--compare", action="store_true", help="Compare basic vs multi_stage")
    args = parser.parse_args()
    
    logger.info("Initializing services...")
    embedding_service = EmbeddingService(
        model_name=config.EMBEDDING_MODEL,
        device=config.EMBEDDING_DEVICE
    )
    vector_store = SimpleVectorStore()
    
    if args.test_data:
        queries, relevant_docs = load_test_data(args.test_data)
    else:
        logger.warning("No test data provided, using sample data")
        queries = [
            "ما هو الذكاء الاصطناعي؟",
            "ما هو التعلم الآلي؟"
        ]
        relevant_docs = [[0], [1]]
    
    evaluator = RetrievalEvaluator(embedding_service, vector_store)
    
    if args.compare:
        results = evaluator.compare_modes(queries, relevant_docs, args.top_k)
        
        print("\n=== Evaluation Results ===\n")
        print("Basic Retrieval:")
        for metric, value in results["basic"].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\nMulti-Stage Retrieval:")
        for metric, value in results["multi_stage"].items():
            print(f"  {metric}: {value:.4f}")
    else:
        results = evaluator.evaluate(
            queries, relevant_docs, config.RETRIEVAL_MODE, args.top_k
        )
        
        print("\n=== Evaluation Results ===\n")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
