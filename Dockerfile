FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Install Python dependencies first for Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install headless Chromium and required Debian Linux shared libraries
RUN playwright install --with-deps chromium

# Copy application source code
COPY app ./app

# Expose default port (Render injects $PORT dynamically at runtime)
EXPOSE 10000

# Run Uvicorn with 1 worker to stay well within Render's 512MB Free RAM limit
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]