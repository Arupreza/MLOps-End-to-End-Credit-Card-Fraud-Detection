# Use a small base with Python 3.10
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: curl for healthcheck, fonts for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl fonts-dejavu tzdata \
 && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# First copy requirements for better layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
# Option A: normal PyPI (will pull CPU wheels for PyTorch on most platforms)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Option B: explicit CPU wheels for PyTorch (uncomment to use PyTorch CPU index)
# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# Copy only the app by default (data/models can be mounted as volumes)
COPY app_streamlit.py /app/app_streamlit.py

# If you want to bake models and data *into* the image, uncomment:
# COPY models/ /app/models/
# COPY Data/ /app/Data/

# Streamlit network config
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

# Healthcheck (simple)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
