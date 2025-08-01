# backend/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your backend source files
COPY backend/ .

# ✅ Copy the model directory into the image
COPY scripts/distilbert-phish/ distilbert-phish/

# ✅ Start the FastAPI server with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



