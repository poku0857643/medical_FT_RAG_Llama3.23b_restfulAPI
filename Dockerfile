# Build Stage
# Use a lightweight Python image
FROM python:3.11-slim AS builder
# Set the working directory
WORKDIR /app
# Install system dependencies for PyMuPDF and FAISS
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Final stage
FROM python:3.11-slim
WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"
# Copy the application code
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Expose the application port (e.g., FastAPI default is 8000)
EXPOSE 8009
# Command to run the application
CMD ["python", "main.py"]
