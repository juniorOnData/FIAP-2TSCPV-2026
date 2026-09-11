FROM python:3.12.3-slim

WORKDIR /app

# =========================================================
# DEPENDÊNCIAS DE COMPILAÇÃO
# =========================================================
# psutil==5.9.0 precisa ser compilado no Python 3.12.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*


# =========================================================
# DEPENDÊNCIAS PYTHON
# =========================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# =========================================================
# APLICAÇÃO
# =========================================================

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]