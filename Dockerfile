# Dockerfile

FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        pkg-config \
        libssl-dev \
        libffi-dev \
        python3-dev \
        rustc \
        cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install transformers==4.38.0 torch accelerate==0.21.0 pandas scikit-learn datasets joblib


CMD ["python", "main.py"]
