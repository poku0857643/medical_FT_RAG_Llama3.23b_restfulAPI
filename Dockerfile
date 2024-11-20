# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libpoppler-cpp-dev \
    libopenblas-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the application port (e.g., FastAPI default is 8000)
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
