FROM python:3.12.3-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 LOG_LEVEL=INFO EMBEDDING_DEVICE=cpu

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --no-compile -r requirements.txt

# Only application code and static assets enter the image. Documents, indexes,
# credentials, tests, and local environment files are excluded from the context.
COPY app/ ./app/
COPY static/ ./static/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
