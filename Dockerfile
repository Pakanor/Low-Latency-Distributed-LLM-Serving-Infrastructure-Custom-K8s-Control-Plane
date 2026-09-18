# Stage 1: Build dependencies & native modules
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools>=61.0 pybind11>=2.10.0 wheel

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-build-isolation --user -e .

FROM python:3.11-slim AS runner

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]