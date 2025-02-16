# Stage 1: Python stage - Use official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for required packages
RUN apt-get update && apt-get install -y \
    git \
    vim \
    npm \
    tesseract-ocr \
    libsqlite3-dev \
    libglib2.0-0 \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN npm install --no-cache-dir prettier@3.4.2

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
