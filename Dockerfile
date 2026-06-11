FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .

# Model is mounted or baked in at deploy time; expose the API.
ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000

CMD ["uvicorn", "hsd.api:app", "--host", "0.0.0.0", "--port", "8000"]
