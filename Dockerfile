#Build stage
FROM python:3.12-slim as builder

WORKDIR /build

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock README.md ./

COPY src ./src

RUN poetry build --format wheel

# Runtime stage

FROM python:3.12-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOUSING_ML_ROOT=/app

WORKDIR /app

COPY --from=builder \
    /build/dist/*.whl \
    /tmp/

RUN pip install \
    --no-cache-dir \
    /tmp/*.whl \
    && rm -f /tmp/*.whl


COPY configs ./configs

COPY artifacts/models/housing_pipeline.joblib \
 ./artifacts/models/housing_pipeline.joblib


EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"

CMD ["python", "-m", "house_ml.api.run"]
 

