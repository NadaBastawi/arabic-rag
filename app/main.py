"""
Main entry point for the Arabic RAG Knowledge Assistant.

This module initializes all services, loads sample documents, and demonstrates
the RAG pipeline by answering a question.
"""

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import SimpleVectorStore
from app.services.rag_pipeline import RAGPipeline


def main():
    """Main function to run the Arabic RAG system."""
    
    print("Initializing Arabic RAG Knowledge Assistant...")
    print("-" * 50)
    
    # Initialize EmbeddingService
    print("Loading embedding model...")
    embedding_service = EmbeddingService()
    print(f"Embedding model loaded: {embedding_service.model_name}")
    
    # Initialize LLMService
    print("Loading LLM model...")
    llm_service = LLMService()
    print(f"LLM model loaded: {llm_service.model_name}")
    
    # Initialize VectorStore
    print("Initializing vector store...")
    vector_store = SimpleVectorStore()
    
    # Sample Arabic documents
    print("Loading sample documents...")
    sample_documents = [
        "الذكاء الاصطناعي هو فرع من فروع علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على محاكاة الذكاء البشري. يشمل الذكاء الاصطناعي التعلم الآلي ومعالجة اللغة الطبيعية والرؤية الحاسوبية.",
        "التعلم الآلي هو نوع من الذكاء الاصطناعي يتيح للأنظمة التعلم من البيانات دون برمجة صريحة. يستخدم خوارزميات رياضية لتحليل البيانات وتحديد الأنماط واتخاذ القرارات.",
        "معالجة اللغة الطبيعية هي مجال من مجالات الذكاء الاصطناعي يركز على التفاعل بين الحاسوب واللغة البشرية. تشمل تطبيقاتها الترجمة الآلية وتحليل المشاعر والمساعدات الذكية.",
        "الرؤية الحاسوبية هي مجال علمي يهدف إلى تمكين الحواسيب من فهم وتفسير المحتوى المرئي. تشمل تطبيقاتها التعرف على الوجوه والكشف عن الأشياء والتحليل الطبي للصور.",
        "الشبكات العصبية الاصطناعية هي نماذج حسابية مستوحاة من بنية الدماغ البشري. تتكون من طبقات من العقد المتصلة التي تعالج المعلومات وتتعلم من خلال التدريب على البيانات."
    ]
    
    # Embed documents and add to vector store
    print("Embedding documents...")
    document_embeddings = embedding_service.embed(sample_documents)
    vector_store.add(sample_documents, document_embeddings)
    print(f"Added {len(sample_documents)} documents to vector store")
    
    # Initialize RAG Pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
        top_k=3
    )
    print("RAG pipeline initialized")
    
    print("-" * 50)
    print("\nProcessing question...")
    
    # Hardcoded Arabic question
    question = "ما هو الذكاء الاصطناعي؟"
    print(f"Question: {question}\n")
    
    # Get answer from RAG pipeline
    try:
        answer = rag_pipeline.answer(question)
        print("Answer:")
        print(answer)
    except Exception as e:
        print(f"Error generating answer: {str(e)}")


if __name__ == "__main__":
    main()

