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

# Install compatible versions based on user's environment
RUN pip install \
    transformers==4.38.0 \
    torch>=2.0.0 \
    accelerate>=1.8.0 \
    pandas>=1.5.0 \
    scikit-learn>=1.0.0 \
    datasets>=2.0.0 \
    joblib>=1.0.0 \
    numpy>=1.21.0

# Create necessary directories
RUN mkdir -p logs results_modern results_classic

CMD ["python", "main.py"]
