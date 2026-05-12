# ============================================================
# Dockerfile — PredictiveTransfo
# Image multi-étapes pour réduire la taille finale
# ============================================================

# ---- Étape 1 : image de base avec les dépendances ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Variables d'environnement Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Outils système nécessaires pour certains packages (ex: xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances dans un virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


# ---- Étape 2 : image finale allégée ----
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copier uniquement libgomp (requis par XGBoost au runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier le virtualenv depuis le builder
COPY --from=builder /opt/venv /opt/venv

# Copier le code de l'application
COPY src/         ./src/
COPY app.py       ./app.py
COPY requirements.txt ./requirements.txt

# Créer les dossiers de données et de modèles (montables en volume)
RUN mkdir -p data/raw data/processed models

# Utilisateur non-root pour la sécurité
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser && \
    chown -R appuser:appuser /app

USER appuser

# Pré-génération des données de démonstration au build (optionnel)
# Commentez cette ligne si vous préférez monter vos propres données via un volume
RUN python -c "from src.data_generator import TransformerDataGenerator; \
               g = TransformerDataGenerator(); df = g.generate(); g.save(df)"

# Exposition du port Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Point d'entrée
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
