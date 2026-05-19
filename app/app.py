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
        self.vector_store.ensure_embedding_dimension(
            self.embedding_service.get_embedding_dimension(),
            reset_on_mismatch=True,
        )
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
  --primary-yellow: #fbbf24;
  --primary-amber: #f59e0b;
  --light-bg: #fffbf0;
  --white: #ffffff;
  --gray-text: #374151;
  --light-border: #fce7b6;
}

body, .gradio-container {
  background: linear-gradient(135deg, #fffbf0 0%, #fef3c7 100%) !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}

.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  background: transparent;
}

/* Header styling */
#header {
  text-align: center;
  margin-bottom: 24px;
  padding: 20px;
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(245, 158, 11, 0.1));
  border-radius: 16px;
  border: 2px solid var(--light-border);
}

#header h1 {
  font-size: 2.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #f59e0b, #fbbf24);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 8px 0;
}

#header p {
  color: var(--gray-text);
  font-size: 1.1rem;
  margin: 8px 0 0 0;
  font-weight: 500;
}

/* Chat container */
#chat-container {
  background: var(--white);
  border: 2px solid var(--light-border);
  border-radius: 16px;
  padding: 20px;
  box-shadow: 0 10px 30px rgba(245, 158, 11, 0.1);
}

/* Chatbot styling */
.gradio-chatbot {
  background: var(--white) !important;
  border: none !important;
  border-radius: 12px !important;
}

.message.user {
  background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  color: white !important;
  margin-bottom: 8px;
}

.message.bot {
  background: #f3f4f6 !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  color: var(--gray-text) !important;
  margin-bottom: 8px;
}

/* Input styling */
.gradio-textbox input, .gradio-textbox textarea {
  border: 2px solid var(--light-border) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  font-size: 1rem !important;
  background: var(--white) !important;
  transition: all 0.3s ease !important;
}

.gradio-textbox input:focus, .gradio-textbox textarea:focus {
  border-color: var(--primary-yellow) !important;
  box-shadow: 0 0 0 3px rgba(251, 191, 36, 0.1) !important;
}

/* Button styling */
.gradio-button {
  background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
  border: none !important;
  color: white !important;
  font-weight: 600 !important;
  border-radius: 10px !important;
  padding: 12px 24px !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 12px rgba(245, 158, 11, 0.2) !important;
}

.gradio-button:hover {
  background: linear-gradient(135deg, #f59e0b, #d97706) !important;
  box-shadow: 0 6px 16px rgba(245, 158, 11, 0.3) !important;
  transform: translateY(-2px);
}

.gradio-button:active {
  transform: translateY(0);
}

/* File input styling */
.gradio-file {
  border: 2px dashed var(--light-border) !important;
  border-radius: 12px !important;
  background: rgba(251, 191, 36, 0.05) !important;
  padding: 20px !important;
}

/* Accordion styling */
.gradio-accordion {
  border: 2px solid var(--light-border) !important;
  border-radius: 12px !important;
  background: var(--white) !important;
  margin-bottom: 12px !important;
}

.gradio-accordion-header {
  background: linear-gradient(90deg, rgba(251, 191, 36, 0.1), transparent) !important;
  border-radius: 10px !important;
}

/* Slider styling */
.gradio-slider {
  accent-color: var(--primary-yellow) !important;
}

/* Checkbox styling */
.gradio-checkbox {
  accent-color: var(--primary-yellow) !important;
}

/* Radio button styling */
.gradio-radio {
  accent-color: var(--primary-yellow) !important;
}

/* Status boxes */
.gradio-textbox[interactive="false"] {
  background: linear-gradient(135deg, rgba(251, 191, 36, 0.05), rgba(245, 158, 11, 0.03)) !important;
}

/* Right panel */
#sidebar {
  background: var(--white);
  border: 2px solid var(--light-border);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 10px 30px rgba(245, 158, 11, 0.1);
}

/* Row and column spacing */
.gradio-row {
  gap: 16px !important;
}

.gradio-column {
  gap: 12px !important;
}

/* Markdown styling */
.gradio-markdown {
  color: var(--gray-text) !important;
}

.gradio-markdown h1, .gradio-markdown h2, .gradio-markdown h3 {
  color: var(--gray-text) !important;
  font-weight: 700 !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(251, 191, 36, 0.1);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb {
  background: var(--primary-yellow);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--primary-amber);
}

/* Label styling */
.gradio-label {
  color: var(--gray-text) !important;
  font-weight: 600 !important;
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
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="yellow"),
        css=CUSTOM_CSS,
    ) as demo:
        with gr.Column(elem_id="header"):
            gr.Markdown("# 💬 Arabic RAG Assistant")
            gr.Markdown(
                "Upload documents and ask questions in Arabic or English. Get answers grounded in your documents."
            )

        with gr.Row():
            with gr.Column(scale=4, elem_id="chat-container"):
                chatbot_ui = gr.Chatbot(
                    label="💭 Conversation",
                    height=560,
                    scale=1,
                    type="messages",
                )

                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask your question in Arabic or English...",
                    lines=2,
                    scale=1,
                )

                with gr.Row(scale=1):
                    submit_btn = gr.Button("📤 Send", variant="primary", scale=1)
                    clear_btn = gr.Button("🔄 New chat", scale=1)

            with gr.Column(scale=2, elem_id="sidebar"):
                with gr.Accordion("⚙️ Retrieval Settings", open=True):
                    retrieval_mode = gr.Radio(
                        ["basic", "multi_stage"],
                        label="Retrieval Mode",
                        value="multi_stage",
                    )
                    top_k_slider = gr.Slider(1, 10, value=5, step=1, label="Top K Results")
                    show_sources = gr.Checkbox(label="Show Sources", value=True)

                with gr.Accordion("📚 Knowledge Base", open=True):
                    gr.Markdown("**Upload Documents**")
                    file_input = gr.File(
                        label="Upload Files (PDF, DOCX, TXT)",
                        file_types=[".pdf", ".docx", ".txt"],
                        file_count="single"
                    )
                    upload_btn = gr.Button("📥 Upload & Index", variant="primary")
                    upload_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    gr.Markdown("**Paste Documents**")
                    docs_textarea = gr.Textbox(
                        label="Paste text (separate by blank line)",
                        lines=6,
                        placeholder="Paste your documents here...",
                    )
                    index_btn = gr.Button("📝 Index Documents", variant="primary")
                    index_status = gr.Textbox(label="Status", interactive=False, lines=2)

                    with gr.Row():
                        docs_list_btn = gr.Button("📋 List Docs", scale=1)
                        stats_btn = gr.Button("📊 Stats", scale=1)

                    kb_output = gr.Markdown()

                with gr.Accordion("🔍 Debug Retrieval", open=False):
                    score_query = gr.Textbox(label="Query", placeholder="Search query...")
                    score_top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
                    score_btn = gr.Button("Search", variant="primary")
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
