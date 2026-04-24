"""
Multi-stage retrieval and re-ranking for Arabic RAG.

This module provides multi-stage retrieval that combines:
1. Semantic search (vector similarity)
2. Lexical overlap (word matching)
3. Root matching (Arabic morphological similarity)
4. BM25 scoring
5. N-gram overlap
6. Exact match bonus
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from app.utils.arabic_morphology import LightArabicAnalyzer


class ArabicReRanker:
    """
    Re-ranker that combines semantic, lexical, and morphological signals.
    
    This re-ranker takes the initial semantic search results and re-ranks
    them using additional Arabic-specific signals:
    - Word overlap (lexical similarity)
    - Root overlap (morphological similarity)
    - BM25 scoring
    - N-gram overlap
    - Exact match bonus
    - Position bonus (earlier mentions)
    
    Attributes:
        weights: Dictionary of weights for each signal.
        analyzer: Light Arabic analyzer for root extraction.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.3,
        lexical_weight: float = 0.2,
        root_weight: float = 0.15,
        bm25_weight: float = 0.15,
        ngram_weight: float = 0.1,
        exact_match_weight: float = 0.1,
        use_position_bonus: bool = True
    ):
        """
        Initialize the re-ranker.
        
        Args:
            semantic_weight: Weight for semantic similarity (0-1).
            lexical_weight: Weight for lexical overlap (0-1).
            root_weight: Weight for root overlap (0-1).
            bm25_weight: Weight for BM25 score (0-1).
            ngram_weight: Weight for n-gram overlap (0-1).
            exact_match_weight: Weight for exact match bonus (0-1).
            use_position_bonus: Whether to boost earlier results.
        """
        self.weights = {
            'semantic': semantic_weight,
            'lexical': lexical_weight,
            'root': root_weight,
            'bm25': bm25_weight,
            'ngram': ngram_weight,
            'exact_match': exact_match_weight
        }
        self.use_position_bonus = use_position_bonus
        self.analyzer = LightArabicAnalyzer()
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Re-rank candidate documents based on combined signals.
        
        Args:
            query: Original query text.
            candidates: List of candidate documents with 'text' and 'score'.
            top_k: Number of top results to return.
            
        Returns:
            Re-ranked list of documents with combined scores.
        """
        if not candidates:
            return []
        
        query = query.strip()
        
        scores = self.analyzer.get_comprehensive_score(query, candidates[0].get('text', ''))
        avg_bm25 = max(scores.get('bm25', 1), 1)
        
        reranked = []
        for idx, doc in enumerate(candidates):
            semantic_score = doc.get('score', 0.0)
            text = doc.get('text', '')
            
            lexical_score = self.analyzer.get_word_overlap(query, text)
            root_score = self.analyzer.get_root_overlap(query, text)
            bm25_score = self.analyzer.get_bm25_score(query, text, avg_doc_len=100.0) / max(avg_bm25, 1)
            ngram_score = self.analyzer.get_ngram_overlap(query, text, 2)
            exact_match = self.analyzer.get_exact_match_bonus(query, text)
            
            combined_score = (
                self.weights['semantic'] * semantic_score +
                self.weights['lexical'] * lexical_score +
                self.weights['root'] * root_score +
                self.weights['bm25'] * bm25_score +
                self.weights['ngram'] * ngram_score +
                self.weights['exact_match'] * exact_match
            )
            
            if self.use_position_bonus and idx < 3:
                position_bonus = 0.05 * (3 - idx)
                combined_score += position_bonus
            
            reranked.append({
                'text': text,
                'score': float(combined_score),
                'semantic_score': float(semantic_score),
                'lexical_score': float(lexical_score),
                'root_score': float(root_score),
                'bm25_score': float(bm25_score),
                'ngram_score': float(ngram_score),
                'exact_match': float(exact_match)
            })
        
        reranked.sort(key=lambda x: x['score'], reverse=True)
        
        return reranked[:top_k]
    
    def get_scores(
        self,
        query: str,
        text: str
    ) -> Dict[str, float]:
        """
        Get individual scores for a query-text pair.
        
        Args:
            query: Query text.
            text: Document text.
            
        Returns:
            Dictionary with individual signal scores.
        """
        return {
            'semantic': 0.0,
            'lexical': self.analyzer.get_word_overlap(query, text),
            'root': self.analyzer.get_root_overlap(query, text)
        }


class MultiStageRetriever:
    """
    Multi-stage retrieval pipeline for Arabic RAG.
    
    Stage 1: Semantic search using embeddings
    Stage 2: Re-ranking using lexical and morphological signals
    
    Attributes:
        embedding_service: Service for generating embeddings.
        vector_store: Vector store for semantic search.
        reranker: Re-ranker for stage 2.
        initial_top_k: Number of candidates to retrieve in stage 1.
    """
    
    def __init__(
        self,
        embedding_service,
        vector_store,
        initial_top_k: int = 20,
        reranker: Optional[ArabicReRanker] = None
    ):
        """
        Initialize the multi-stage retriever.
        
        Args:
            embedding_service: Embedding service for query encoding.
            vector_store: Vector store for semantic search.
            initial_top_k: Number of candidates to retrieve in stage 1.
            reranker: Re-ranker instance (creates default if None).
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.initial_top_k = initial_top_k
        self.reranker = reranker or ArabicReRanker()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve relevant documents using multi-stage approach.
        
        Args:
            query: Query text.
            top_k: Final number of results to return.
            
        Returns:
            List of relevant documents with scores.
        """
        query_embedding = self.embedding_service.embed(query)
        
        candidates = self.vector_store.search(
            query_embedding,
            top_k=self.initial_top_k
        )
        
        if not candidates:
            return []
        
        reranked = self.reranker.rerank(query, candidates, top_k)
        
        return reranked
    
    def retrieve_with_stages(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve documents and return results from each stage.
        
        Args:
            query: Query text.
            top_k: Final number of results.
            
        Returns:
            Dictionary with 'stage1' (semantic) and 'stage2' (reranked) results.
        """
        query_embedding = self.embedding_service.embed(query)
        
        stage1_results = self.vector_store.search(
            query_embedding,
            top_k=self.initial_top_k
        )
        
        stage2_results = self.reranker.rerank(
            query,
            stage1_results,
            top_k
        )
        
        return {
            'stage1': stage1_results,
            'stage2': stage2_results
        }


class AdaptiveRetrieval:
    """
    Adaptive retrieval that adjusts strategy based on query characteristics.
    
    Strategies:
    - High lexical overlap: Boost root matching
    - Abstract query: Use semantic search primarily
    - Specific entities: Combine with entity matching
    """
    
    def __init__(
        self,
        embedding_service,
        vector_store,
        base_reranker: Optional[ArabicReRanker] = None
    ):
        """
        Initialize adaptive retriever.
        
        Args:
            embedding_service: Embedding service.
            vector_store: Vector store.
            base_reranker: Base re-ranker to adapt.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.base_reranker = base_reranker or ArabicReRanker()
        self.analyzer = LightArabicAnalyzer()
    
    def _analyze_query(self, query: str) -> Dict[str, any]:
        """
        Analyze query characteristics to determine best strategy.
        
        Args:
            query: Query text.
            
        Returns:
            Dictionary with query analysis.
        """
        words = query.split()
        
        return {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'has_question_mark': '؟' in query or '?' in query,
            'roots': self.analyzer.extract_roots(query)
        }
    
    def _get_adaptive_weights(self, query_analysis: Dict) -> Tuple[float, float, float]:
        """
        Get adaptive weights based on query analysis.
        
        Args:
            query_analysis: Query analysis results.
            
        Returns:
            Tuple of (semantic_weight, lexical_weight, root_weight).
        """
        word_count = query_analysis['word_count']
        
        if word_count <= 2:
            return (0.3, 0.3, 0.4)
        elif word_count <= 5:
            return (0.5, 0.3, 0.2)
        else:
            return (0.6, 0.25, 0.15)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Retrieve with adaptive strategy selection.
        
        Args:
            query: Query text.
            top_k: Number of results.
            
        Returns:
            List of relevant documents.
        """
        query_analysis = self._analyze_query(query)
        
        semantic_w, lexical_w, root_w = self._get_adaptive_weights(query_analysis)
        
        adaptive_reranker = ArabicReRanker(
            semantic_weight=semantic_w,
            lexical_weight=lexical_w,
            root_weight=root_w
        )
        
        retriever = MultiStageRetriever(
            self.embedding_service,
            self.vector_store,
            reranker=adaptive_reranker
        )
        
        return retriever.retrieve(query, top_k)
