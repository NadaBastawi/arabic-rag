"""
Minimal tests for the RAG pipeline.

Tests document addition, context retrieval, and answer generation.
"""

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import SimpleVectorStore
from app.services.rag_pipeline import RAGPipeline


class MockLLMService:
    """Mock LLM service that returns a simple response."""
    
    def generate(self, prompt: str) -> str:
        """Return a mock answer."""
        return "هذه إجابة تجريبية من النموذج اللغوي."


def test_documents_can_be_added():
    """Test that documents can be added to the vector store."""
    vector_store = SimpleVectorStore()
    embedding_service = EmbeddingService()
    
    # Sample documents
    documents = [
        "الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب.",
        "التعلم الآلي يستخدم خوارزميات لتحليل البيانات."
    ]
    
    # Embed and add documents
    embeddings = embedding_service.embed(documents)
    vector_store.add(documents, embeddings)
    
    # Assert documents were added
    assert len(vector_store.texts) == 2
    assert vector_store.embeddings is not None
    assert vector_store.embeddings.shape[0] == 2


def test_query_returns_non_empty_context():
    """Test that a query returns non-empty context from the vector store."""
    vector_store = SimpleVectorStore()
    embedding_service = EmbeddingService()
    
    # Add sample documents
    documents = [
        "الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب يهدف إلى إنشاء أنظمة ذكية.",
        "التعلم الآلي هو نوع من الذكاء الاصطناعي يتيح للأنظمة التعلم من البيانات.",
        "معالجة اللغة الطبيعية تركز على التفاعل بين الحاسوب واللغة البشرية."
    ]
    
    embeddings = embedding_service.embed(documents)
    vector_store.add(documents, embeddings)
    
    # Query
    question = "ما هو الذكاء الاصطناعي؟"
    question_embedding = embedding_service.embed(question)
    
    # Search
    results = vector_store.search(question_embedding, top_k=2)
    
    # Assert non-empty results
    assert len(results) > 0
    assert isinstance(results[0], dict)
    assert 'text' in results[0]
    assert len(results[0]['text']) > 0


def test_pipeline_returns_string_answer():
    """Test that the RAG pipeline returns a string answer."""
    # Initialize services
    embedding_service = EmbeddingService()
    llm_service = MockLLMService()
    vector_store = SimpleVectorStore()
    
    # Add documents
    documents = [
        "الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب.",
        "التعلم الآلي يستخدم خوارزميات لتحليل البيانات."
    ]
    
    embeddings = embedding_service.embed(documents)
    vector_store.add(documents, embeddings)
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
        top_k=2
    )
    
    # Get answer
    question = "ما هو الذكاء الاصطناعي؟"
    answer = pipeline.answer(question)
    
    # Assert answer is a non-empty string
    assert isinstance(answer, str)
    assert len(answer) > 0


if __name__ == "__main__":
    print("Running tests...")
    
    print("\n1. Testing document addition...")
    test_documents_can_be_added()
    print("   ✓ Documents can be added")
    
    print("\n2. Testing query returns context...")
    test_query_returns_non_empty_context()
    print("   ✓ Query returns non-empty context")
    
    print("\n3. Testing pipeline returns answer...")
    test_pipeline_returns_string_answer()
    print("   ✓ Pipeline returns string answer")
    
    print("\nAll tests passed!")

