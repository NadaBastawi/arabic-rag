"""Regression tests for Arabic prompt and context text."""

from app.services.rag_pipeline import RAGPipeline


def test_prompt_preserves_arabic_instruction_question_and_context():
    question = "ما هي عاصمة مصر؟"
    context = "نص"
    pipeline = RAGPipeline(None, None, None)

    prompt = pipeline._construct_prompt(question, context)

    assert prompt.startswith("استخدم المعلومات التالية من قاعدة المعرفة للإجابة على السؤال.")
    assert f"المعلومات من قاعدة المعرفة:\n{context}" in prompt
    assert f"السؤال: {question}" in prompt
    assert prompt.endswith("الإجابة:")


def test_answer_uses_arabic_context_labels():
    class Embeddings:
        def embed(self, question):
            return question

    class Store:
        def __init__(self, documents):
            self.documents = documents

        def search(self, embedding, top_k):
            return self.documents

    class LLM:
        def generate(self, prompt):
            return prompt

    empty = RAGPipeline(Embeddings(), LLM(), Store([]))
    populated = RAGPipeline(Embeddings(), LLM(), Store([{"text": "نص"}]))

    assert "لا توجد معلومات متاحة في قاعدة المعرفة." in empty.answer("سؤال")
    assert "المستند 1: نص" in populated.answer("سؤال")
