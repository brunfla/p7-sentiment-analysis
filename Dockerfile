# Étape 1 : Utiliser une image de base Python légère
FROM python:3.9-slim

# Étape 2 : Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends build-essential locales locales-all && \ 
    rm -rf /var/lib/apt/lists/*

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV TF_USE_LEGACY_KERAS=1 

# Étape 2 : Définir le répertoire de travail
WORKDIR /app

# Étape 3 : Copier les fichiers de l'application dans le conteneur
COPY data/output/trained_models/distilbert-base-uncased/final_model /app/final_model 
COPY src/deploy/api.py  /app/api.py

# Étape 4 : Installer les dépendances
RUN pip install flask==3.0.3
RUN pip install gunicorn==22.0.0
RUN pip install ktrain==0.41.4
RUN pip install transformers==4.18.0
RUN pip install tf_keras==2.18.0
RUN pip install opencensus==0.11.4
RUN pip install opencensus-ext-azure==1.1.14
RUN pip install opencensus-ext-flask==0.8.2
#RUN pip install tensorflow-cpu==2.18.0

# Étape 5 : Exposer le port utilisé par l'application
EXPOSE 8000

# Étape 6 : Démarrer l'application avec Gunicorn pour la production
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:8000", "api:app"]
