# Étape 1 : Utiliser une image de base Python légère
FROM python:3.9-slim

# Étape 2 : Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends build-essential locales locales-all && \ 
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Copier les fichiers de l'application dans le conteneur
COPY logreg_model.onnx /app/logreg_model.onnx
COPY app.py /app/app.py 
COPY requirements.txt /app/requirements.txt 

# Étape 4 : Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --upgrade onnxruntime

# Étape 5 : Exposer le port utilisé par l'application
EXPOSE 8000

# Étape 6 : Démarrer l'application avec Gunicorn pour la production
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]
