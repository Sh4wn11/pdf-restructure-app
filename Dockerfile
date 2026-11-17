FROM python:3.12-slim-bookworm

# Suppress .pyc files, buffer logs, silence pip root warning, and set browser path
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Refresh Debian Bookworm apt index, install Chromium + OS libraries, and clean cache
RUN apt-get update && \
    playwright install --with-deps chromium && \
    rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY app ./app

EXPOSE 10000

# Start FastAPI bound to Render's dynamic port
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]