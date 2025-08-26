# Use a small base with Python 3.10
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl fonts-dejavu tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements first for cache efficiency
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
# (If you need CPU-only torch wheels, use the torch CPU index instead.)

# App code
COPY app_streamlit.py /app/app_streamlit.py

# ✅ Bake models (must include models/model_info.json and your .pth)
COPY models/ /app/models/

# ✅ (Optional) Bake a default CSV so “Run Evaluation” works without upload/mount
#    If you keep your code using TEST_DATA_PATH, set it here:
# COPY Data/processed/creditcard_processed_test.csv /app/Data/processed/creditcard_processed_test.csv
# ENV TEST_DATA_PATH=/app/Data/processed/creditcard_processed_test.csv

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]