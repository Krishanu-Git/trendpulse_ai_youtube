# Use a lightweight Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install OS dependencies (for wordcloud, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Download required NLTK datasets
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('vader_lexicon')"

# Streamlit UI port
EXPOSE 8501

# Run Streamlit App
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]