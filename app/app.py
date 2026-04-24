"""
Gradio UI for Arabic RAG chatbot.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from app.config import config
from app.services.embedding_service import EmbeddingService
from app.services.faiss_store import FAISSVectorStore
from app.services.llm_service import LLMService
from app.services.rag_pipeline import RAGPipeline
from app.utils.document_loader import DocumentLoader
from app.utils.errors import InputValidator, ValidationError

logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

Message = Dict[str, Any]


class ArabicRAGChatbot:
    """Simple chatbot wrapper for Arabic RAG."""

    def __init__(self):
        self.embedding_service = EmbeddingService(
            model_name=config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )
        self.llm_service = LLMService(
            model=config.LLM_MODEL,
            api_key=config.LLM_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        self.vector_store = FAISSVectorStore(index_path=config.INDEX_PATH)
        self.document_loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
        self.document_registry: Dict[str, Dict[str, Any]] = {}

        self.rag_pipeline = RAGPipeline(
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
            vector_store=self.vector_store,
            top_k=config.RETRIEVAL_TOP_K,
            retrieval_mode=config.RETRIEVAL_MODE,
            initial_top_k=config.RETRIEVAL_INITIAL_K,
            use_adaptive=config.USE_ADAPTIVE,
        )

        self._hydrate_registry_from_store()

    def _hydrate_registry_from_store(self) -> None:
        """Populate registry from persisted vector store metadata."""
        grouped: Dict[str, Dict[str, Any]] = {}

        for item in self.vector_store.metadata:
            doc_id = item.get("document_id")
            if not doc_id:
                continue

            if doc_id not in grouped:
                grouped[doc_id] = {
                    "source": item.get("source", "unknown"),
                    "chunk_count": 0,
                    "added_at": datetime.now().isoformat(),
                }
            grouped[doc_id]["chunk_count"] += 1

        self.document_registry = grouped

    @staticmethod
    def _ensure_messages(history: Optional[List[Any]]) -> List[Message]:
        """Normalize history into Gradio messages format."""
        if not history:
            return []

        normalized: List[Message] = []

        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                normalized.append({"role": item["role"], "content": item["content"]})
                continue

            if isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    normalized.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    normalized.append({"role": "assistant", "content": str(assistant_msg)})

        return normalized

    def _append_sources(self, answer: str, docs: List[Dict[str, Any]]) -> str:
        if not docs:
            return answer

        lines = ["", "Sources:"]
        for idx, doc in enumerate(docs[:3], 1):
            source = doc.get("source", "unknown")
            doc_id = doc.get("document_id", "")
            score = doc.get("score", 0.0)
            preview = doc.get("text", "").replace("\n", " ").strip()[:180]
            lines.append(
                f"{idx}. {source} | score={score:.3f}" + (f" | id={doc_id}" if doc_id else "")
            )
            if preview:
                lines.append(f"   {preview}...")

        return answer + "\n" + "\n".join(lines)

    def index_documents(self, texts: List[str]) -> str:
        if not texts:
            return "No documents found to index."

        try:
            embeddings = self.embedding_service.embed(texts)
            doc_ids = [f"manual_{i}" for i in range(len(texts))]

            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                document_ids=doc_ids,
                sources=["manual_input"] * len(texts),
            )

            for doc_id in doc_ids:
                self.document_registry[doc_id] = {
                    "source": "manual_input",
                    "added_at": datetime.now().isoformat(),
                    "chunk_count": 1,
                }

            return f"Indexed {len(texts)} documents successfully."
        except Exception as exc:
            logger.error("Indexing error: %s", str(exc))
            return f"Indexing failed: {str(exc)}"

    def upload_document(self, file_path: str) -> str:
        if not file_path:
            return "No file selected."

        try:
            chunks = self.document_loader.load_file(file_path)
            if not chunks:
                return f"Could not extract text from {Path(file_path).name}."

            texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_service.embed(texts)

            doc_id = Path(file_path).stem
            sources = [chunk.source for chunk in chunks]

            self.vector_store.add_documents(
                texts=texts,
                embeddings=embeddings,
                document_ids=[doc_id] * len(chunks),
                sources=sources,
            )

            self.document_registry[doc_id] = {
                "source": Path(file_path).name,
                "added_at": datetime.now().isoformat(),
                "chunk_count": len(chunks),
            }

            return f"Uploaded {Path(file_path).name} with {len(chunks)} chunks."
        except Exception as exc:
            logger.error("Upload error: %s", str(exc))
            return f"Upload failed: {str(exc)}"

    def list_documents(self) -> str:
        if not self.document_registry:
            return "No documents indexed yet."

        lines = ["Indexed Documents:"]
        for doc_id, info in self.document_registry.items():
            lines.append(
                f"- {doc_id} | chunks={info['chunk_count']} | source={info['source']}"
            )
        return "\n".join(lines)

    def get_document_stats(self) -> str:
        stats = self.vector_store.get_stats()
        return (
            "System Stats:\n"
            f"- total_documents: {stats.get('total_documents')}\n"
            f"- total_vectors: {stats.get('total_vectors')}\n"
            f"- index_type: {stats.get('index_type')}\n"
            f"- embedding_dim: {stats.get('embedding_dim')}"
        )

    def get_retrieval_scores(self, query: str, top_k: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "Query cannot be empty."

        try:
            docs = self.rag_pipeline.retrieve_with_scores(query)
            docs = docs[:top_k]
            if not docs:
                return "No retrieval results."

            lines = ["Retrieval Scores:"]
            for idx, doc in enumerate(docs, 1):
                lines.append(
                    f"{idx}. score={doc.get('score', 0.0):.4f} | text={doc.get('text', '')[:120]}"
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.error("Scores error: %s", str(exc))
            return f"Could not get scores: {str(exc)}"

    def chat(
        self,
        message: str,
        history: Optional[List[Any]],
        retrieval_mode: str = "multi_stage",
        top_k: int = 5,
        show_sources: bool = True,
    ):
        messages = self._ensure_messages(history)

        if not self.document_registry:
            messages.append(
                {
                    "role": "assistant",
                    "content": "Please index or upload documents first.",
                }
            )
            return "", messages

        if not message or not message.strip():
            return "", messages

        try:
            user_message = InputValidator.validate_query(message, max_length=1000)

            original_mode = self.rag_pipeline.retrieval_mode
            original_top_k = self.rag_pipeline.top_k

            self.rag_pipeline.retrieval_mode = retrieval_mode
            self.rag_pipeline.top_k = top_k

            answer = self.rag_pipeline.answer(user_message)

            docs: List[Dict[str, Any]] = []
            if retrieval_mode == "multi_stage" and self.rag_pipeline.multi_stage_retriever:
                docs = self.rag_pipeline.multi_stage_retriever.retrieve(user_message, top_k=top_k)
            else:
                query_embedding = self.embedding_service.embed(user_message)
                docs = self.vector_store.search(query_embedding, top_k=top_k)

            if show_sources:
                answer = self._append_sources(answer, docs)

            self.rag_pipeline.retrieval_mode = original_mode
            self.rag_pipeline.top_k = original_top_k

            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": answer})
            return "", messages

        except ValidationError as exc:
            messages.append({"role": "assistant", "content": f"Input error: {str(exc)}"})
            return "", messages
        except Exception as exc:
            logger.error("Chat error: %s", str(exc))
            messages.append({"role": "assistant", "content": f"Chat error: {str(exc)}"})
            return "", messages


CUSTOM_CSS = """
:root {
  --bg: #f7f7f8;
  --panel: #ffffff;
  --border: #e5e7eb;
  --text: #111827;
  --muted: #6b7280;
}
.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
  background: var(--bg);
}
#chat-shell {
  border: 1px solid var(--border);
  border-radius: 14px;
  background: var(--panel);
  padding: 10px;
}
#header h1 {
  font-size: 1.4rem;
  margin-bottom: 0.2rem;
}
#header p {
  color: var(--muted);
  margin-top: 0;
}
"""


def _extract_uploaded_path(uploaded: Any) -> str:
    if uploaded is None:
        return ""

    if isinstance(uploaded, str):
        return uploaded

    file_name = getattr(uploaded, "name", None)
    if isinstance(file_name, str):
        return file_name

    return ""


def create_ui() -> gr.Blocks:
    chatbot = ArabicRAGChatbot()

    with gr.Blocks(
        title="Arabic RAG Chatbot",
        theme=gr.themes.Soft(primary_hue="gray", secondary_hue="slate"),
        css=CUSTOM_CSS,
    ) as demo:
        with gr.Column(elem_id="header"):
            gr.Markdown("# Arabic RAG Assistant")
            gr.Markdown(
                "Simple chat experience. Upload/index docs, then ask questions with grounded answers."
            )

        with gr.Row():
            with gr.Column(scale=4, elem_id="chat-shell"):
                chatbot_ui = gr.Chatbot(
                    label="Conversation",
                    height=560,
                )

                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask in Arabic or English...",
                    lines=2,
                )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("New chat")

            with gr.Column(scale=2):
                with gr.Accordion("Retrieval Settings", open=True):
                    retrieval_mode = gr.Radio(
                        ["basic", "multi_stage"],
                        label="Mode",
                        value="multi_stage",
                    )
                    top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    show_sources = gr.Checkbox(label="Show sources", value=True)

                with gr.Accordion("Knowledge Base", open=True):
                    file_input = gr.File(label="Upload PDF/TXT", file_types=[".pdf", ".txt"])
                    upload_btn = gr.Button("Upload and index")
                    upload_status = gr.Textbox(label="Upload status", interactive=False)

                    docs_textarea = gr.Textbox(
                        label="Paste documents (separate by blank line)",
                        lines=8,
                    )
                    index_btn = gr.Button("Index pasted documents")
                    index_status = gr.Textbox(label="Index status", interactive=False)

                    with gr.Row():
                        docs_list_btn = gr.Button("List docs")
                        stats_btn = gr.Button("Stats")

                    kb_output = gr.Markdown()

                with gr.Accordion("Debug Retrieval", open=False):
                    score_query = gr.Textbox(label="Query")
                    score_top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    score_btn = gr.Button("Get scores")
                    score_output = gr.Markdown()

        submit_btn.click(
            chatbot.chat,
            inputs=[msg_input, chatbot_ui, retrieval_mode, top_k_slider, show_sources],
            outputs=[msg_input, chatbot_ui],
        )

        msg_input.submit(
            chatbot.chat,
            inputs=[msg_input, chatbot_ui, retrieval_mode, top_k_slider, show_sources],
            outputs=[msg_input, chatbot_ui],
        )

        clear_btn.click(lambda: ("", []), outputs=[msg_input, chatbot_ui])

        upload_btn.click(
            lambda f: chatbot.upload_document(_extract_uploaded_path(f)),
            inputs=[file_input],
            outputs=[upload_status],
        )

        index_btn.click(
            lambda txt: chatbot.index_documents([chunk.strip() for chunk in txt.split("\n\n") if chunk.strip()]),
            inputs=[docs_textarea],
            outputs=[index_status],
        )

        docs_list_btn.click(chatbot.list_documents, outputs=[kb_output])
        stats_btn.click(chatbot.get_document_stats, outputs=[kb_output])

        score_btn.click(
            chatbot.get_retrieval_scores,
            inputs=[score_query, score_top_k],
            outputs=[score_output],
        )

    return demo


def main() -> None:
    demo = create_ui()
    demo.launch(server_port=config.GRADIO_PORT, share=config.GRADIO_SHARE)


if __name__ == "__main__":
    main()
