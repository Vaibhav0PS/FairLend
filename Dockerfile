FROM python:3.11-slim

# Lambda Web Adapter: lets this same image run on AWS Lambda. It forwards Lambda
# events to uvicorn as HTTP requests. Outside Lambda it does nothing.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health

# XGBoost needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Folder with the model files to bake in. Default: the committed local model.
# The deploy pipeline passes the folder holding the registry's "champion" model.
ARG MODEL_SRC=artifacts

COPY credit_risk/ credit_risk/
COPY static/ static/
COPY ${MODEL_SRC}/ artifacts/
COPY main.py .

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
