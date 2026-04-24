# Arabic RAG System - Complete Setup and Deployment Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- For GPU support: CUDA 11.8+

### Local Setup

1. **Clone and setup virtual environment:**
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
source .venv/bin/activate      # macOS/Linux
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

4. **Run the system:**

**Option A: Both API and UI (Recommended)**
```bash
# Terminal 1: Start FastAPI
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Gradio UI
python -m gradio app.app
```

**Option B: Gradio UI Only**
```bash
python -m gradio app.app
```

**Option C: FastAPI Only**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker Deployment

### Deploy with Docker Compose (Easiest)

```bash
# 1. Set your LLM API key
export LLM_API_KEY=your-api-key-here

# 2. Build and run
docker-compose up -d

# 3. Access services
# - Gradio UI: http://localhost:7860
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Deploy with Docker CLI

```bash
# Build image
docker build -t arabic-rag:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -p 7860:7860 \
  -e LLM_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name arabic-rag \
  arabic-rag:latest
```

---

## 📋 Configuration

### Environment Variables

Create `.env` file based on `.env.example`:

```env
# Embedding Model
EMBEDDING_MODEL=intfloat/multilingual-e5-large
EMBEDDING_DEVICE=auto

# LLM Setup
LLM_MODEL=gemini-2.0-flash
LLM_API_KEY=your-api-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=2048

# Retrieval
RETRIEVAL_MODE=multi_stage
RETRIEVAL_TOP_K=5
USE_ADAPTIVE=true

# Logging
LOG_LEVEL=INFO

# Ports
API_PORT=8000
GRADIO_PORT=7860
```

### Supported LLM Models

- **Google Gemini:** `gemini-2.0-flash`, `gemini-2.0-flash-exp`
- **OpenAI:** `gpt-4o`, `gpt-4o-mini`
- **Local:** Any Hugging Face model ID

---

## 🔧 API Documentation

### Health Check
```bash
curl http://localhost:8000/health
```

### Index Documents
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "الذكاء الاصطناعي هو...",
      "التعلم الآلي هو..."
    ]
  }'
```

### Query Documents
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "ما هو الذكاء الاصطناعي؟",
    "top_k": 5
  }'
```

### Retrieve Documents
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ما هو الذكاء الاصطناعي؟",
    "top_k": 5
  }'
```

### Add Single Document
```bash
curl -X POST http://localhost:8000/documents/add \
  -H "Content-Type: application/json" \
  -d '{
    "text": "المستند النصي...",
    "document_id": "doc_1",
    "source": "manual"
  }'
```

### Upload File
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"
```

### List Documents
```bash
curl http://localhost:8000/documents
```

### Delete Document
```bash
curl -X DELETE http://localhost:8000/documents/doc_1
```

### Clear All Documents
```bash
curl -X POST http://localhost:8000/documents/clear
```

### Get Statistics
```bash
curl http://localhost:8000/stats
```

### View Configuration
```bash
curl http://localhost:8000/config
```

**Full API Documentation:** Visit `http://localhost:8000/docs` after starting the server

---

## 📊 System Architecture

```
Arabic RAG System
├── Embedding Service (intfloat/multilingual-e5-large)
├── Vector Store (FAISS with persistence)
├── Multi-Stage Retrieval
│   ├── Semantic search
│   ├── Lexical matching
│   ├── Arabic root matching
│   ├── BM25 scoring
│   ├── N-gram overlap
│   └── Exact match bonus
├── LLM Service (Gemini/OpenAI/Local)
├── Caching Layer
│   ├── Embedding cache
│   └── Query result cache
└── RAG Pipeline
    └── Answer generation with grounding
```

---

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Integration Tests
```bash
pytest tests/test_integration.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=app --cov-report=html
```

### Run Specific Test
```bash
pytest tests/test_integration.py::TestDocumentLoader::test_load_text_document -v
```

---

## 📁 Project Structure

```
arabic-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── app.py                  # Gradio UI
│   ├── config.py               # Configuration
│   ├── build_index.py          # Index building script
│   ├── eval.py                 # Evaluation metrics
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── rag_pipeline.py
│   │   ├── vector_store.py
│   │   ├── faiss_store.py      # Persistent FAISS store
│   │   ├── multi_stage_retrieval.py
│   │   └── cache.py            # Caching layer
│   └── utils/
│       ├── arabic_preprocessing.py
│       ├── arabic_morphology.py
│       ├── document_loader.py   # PDF/text loading
│       ├── errors.py            # Error handling
│       └── logging_config.py    # Logging setup
├── tests/
│   ├── test_rag_pipeline.py    # Basic tests
│   └── test_integration.py     # Integration tests
├── data/
│   ├── index/                  # FAISS index storage
│   └── documents/              # Document storage
├── logs/                       # Application logs
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker Compose config
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

---

## 🎯 Key Features

✅ **Arabic-Aware Processing**
- Letter normalization and diacritic removal
- Morphological analysis with root extraction
- Multilingual embeddings support

✅ **Advanced Retrieval**
- Multi-stage retrieval with re-ranking
- Semantic + lexical + morphological scoring
- Adaptive retrieval with automatic tuning

✅ **Document Management**
- PDF and text file support
- Automatic chunking with overlap
- Document metadata tracking

✅ **Persistent Storage**
- FAISS-based vector store with disk persistence
- Metadata and indexing preservation
- Efficient similarity search

✅ **Performance**
- Embedding and query result caching
- Batch processing support
- Optimized for Arabic text

✅ **Production Ready**
- FastAPI REST API with full documentation
- Gradio web UI with RTL support
- Comprehensive error handling
- Structured logging with rotation
- Docker deployment support

---

## 🔍 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:** Set `EMBEDDING_DEVICE=cpu` in `.env`

### Issue: "Failed to load embedding model"
**Solution:** Check internet connection, model will download on first use

### Issue: "No API key provided"
**Solution:** Set `LLM_API_KEY` in `.env` file

### Issue: "FAISS index not found"
**Solution:** Index builds automatically on first document addition

### Issue: Docker port already in use
**Solution:** Change ports in docker-compose.yml:
```yaml
ports:
  - "8001:8000"  # Change 8001 to desired port
  - "7861:7860"
```

---

## 📚 Dependencies

- **sentence-transformers:** Embeddings
- **torch:** Deep learning framework
- **faiss-cpu:** Vector search (GPU version: faiss-gpu)
- **fastapi/uvicorn:** REST API server
- **gradio:** Web UI framework
- **PyPDF2:** PDF extraction
- **camel-tools:** Arabic morphology
- **pydantic:** Data validation
- **python-dotenv:** Environment configuration

---

## 🚀 Performance Optimization

### For Large Documents
```python
# Increase chunk size in .env
CHUNK_SIZE=1000      # Larger chunks
CHUNK_OVERLAP=100    # More overlap for context
```

### For GPU (CUDA)
```bash
pip install faiss-gpu
export EMBEDDING_DEVICE=cuda:0
```

### For High Concurrency
- Use Docker/Kubernetes for scaling
- Increase caching TTLs
- Monitor memory usage

---

## 📝 License

This project is part of E-JUST research initiative.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Check logs in `./logs` directory

---

**Last Updated:** April 2026
