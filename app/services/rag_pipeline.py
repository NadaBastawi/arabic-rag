"""
RAG pipeline for question answering.

This module provides the RAGPipeline class that combines embedding retrieval
with LLM generation to answer questions based on a knowledge base.
"""

from typing import List, Optional, Protocol
import numpy as np
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService


class VectorStore(Protocol):
    """
    Protocol for vector store implementations.
    
    This protocol defines the interface that vector stores must implement
    to be compatible with the RAG pipeline. Any vector store implementation
    (e.g., FAISS, ChromaDB, Pinecone) should follow this interface.
    """
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of top results to return.
            
        Returns:
            List of dictionaries, each containing at least:
            - 'text': The document text
            - 'score': Similarity score (optional)
            - Additional metadata (optional)
        """
        ...


class SimpleVectorStore:
    """
    Simple in-memory vector store implementation.
    
    This is a placeholder implementation for development and testing.
    For production, replace with a proper vector database like FAISS,
    ChromaDB, Pinecone, or Weaviate.
    
    Attributes:
        texts: List of document texts.
        embeddings: Numpy array of document embeddings.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray
    ) -> None:
        """
        Add documents and their embeddings to the store.
        
        Args:
            texts: List of document texts.
            embeddings: Numpy array of embeddings (num_docs, embedding_dim).
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        self.texts.extend(texts)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        """
        Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of top results to return.
            
        Returns:
            List of dictionaries with 'text' and 'score' keys.
        """
        if self.embeddings is None or len(self.texts) == 0:
            return []
        
        # Compute cosine similarity
        query_embedding = query_embedding.reshape(1, -1)
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'score': float(similarities[idx])
            })
        
        return results


class RAGPipeline:
    """
    RAG pipeline for question answering using retrieval and generation.
    
    This class orchestrates the RAG process:
    1. Embeds the user's question
    2. Retrieves relevant documents from the vector store
    3. Constructs a context from retrieved documents
    4. Generates an answer using the LLM with the context
    
    The pipeline is designed to be modular, allowing easy swapping of
    embedding services, LLM services, and vector stores.
    
    Attributes:
        embedding_service: Service for generating embeddings.
        llm_service: Service for generating text responses.
        vector_store: Vector store for document retrieval.
        top_k: Number of documents to retrieve for context.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        vector_store: VectorStore,
        top_k: int = 5
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_service: Instance of EmbeddingService for generating embeddings.
            llm_service: Instance of LLMService for generating answers.
            vector_store: Vector store instance implementing the VectorStore protocol.
            top_k: Number of top documents to retrieve for context. Defaults to 5.
        """
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.top_k = top_k
    
    def answer(self, question: str) -> str:
        """
        Answer a question using the RAG pipeline.
        
        This method performs the complete RAG process:
        1. Embeds the question
        2. Searches the vector store for relevant documents
        3. Constructs a context string from retrieved documents
        4. Generates an answer using the LLM with the context
        
        Args:
            question: The question to answer (in Arabic or other supported language).
            
        Returns:
            Generated answer string based on retrieved context.
            
        Raises:
            ValueError: If question is empty or invalid.
            RuntimeError: If any step of the pipeline fails.
            
        Example:
            >>> pipeline = RAGPipeline(embedding_service, llm_service, vector_store)
            >>> answer = pipeline.answer("ما هو الذكاء الاصطناعي؟")
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        try:
            # Step 1: Embed the question
            question_embedding = self.embedding_service.embed(question)
            
            # Step 2: Search for relevant documents
            retrieved_docs = self.vector_store.search(
                question_embedding,
                top_k=self.top_k
            )
            
            # Step 3: Construct context from retrieved documents
            if not retrieved_docs:
                context = "لا توجد معلومات متاحة في قاعدة المعرفة."
            else:
                context_parts = [
                    f"المستند {i+1}: {doc['text']}"
                    for i, doc in enumerate(retrieved_docs)
                ]
                context = "\n\n".join(context_parts)
            
            # Step 4: Construct prompt with context and question
            prompt = self._construct_prompt(question, context)
            
            # Step 5: Generate answer using LLM
            answer = self.llm_service.generate(prompt)
            
            return answer
            
        except ValueError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to generate answer: {str(e)}"
            ) from e
    
    def _construct_prompt(self, question: str, context: str) -> str:
        """
        Construct a prompt for the LLM with context and question.
        
        This method formats the context and question into a prompt that
        instructs the LLM to answer based on the provided context.
        
        Args:
            question: The user's question.
            context: Retrieved context documents.
            
        Returns:
            Formatted prompt string for the LLM.
        """
        prompt = f"""استخدم المعلومات التالية من قاعدة المعرفة للإجابة على السؤال. إذا لم تجد الإجابة في المعلومات المقدمة، قل أنك لا تعرف.

المعلومات من قاعدة المعرفة:
{context}

السؤال: {question}

الإجابة:"""
        
        return prompt

