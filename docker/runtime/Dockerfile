FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir .

EXPOSE 80

# Default: all-in-one mode (API + workers)
# Override with: ["llamphouse", "serve", "--no-workers"] or ["llamphouse", "worker"]
ENTRYPOINT ["llamphouse"]
CMD ["serve", "app:app", "--host", "0.0.0.0", "--port", "80"]
