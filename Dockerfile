FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "\
import nltk; \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('wordnet'); \
nltk.download('stopwords'); \
nltk.download('vader_lexicon'); \
nltk.download('averaged_perceptron_tagger')"

# Copy application code
COPY app/ ./app/

# Create required directories
RUN mkdir -p app/models app/data app/reports/drift app/static app/templates

WORKDIR /app/app

EXPOSE 7860

# HuggingFace Spaces uses port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
