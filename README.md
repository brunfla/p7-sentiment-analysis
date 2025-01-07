# Quand le Machine Learning rencontre la philosophie DevOps

## Introduction

Les projets de Machine Learning (ML) sont souvent comparés à des expériences scientifiques : on explore, on teste, on ajuste. Une sorte de laboratoire où chaque idée est mise à l’épreuve. Mais cette approche, bien qu’efficace au début, montre vite ses limites quand il s’agit de passer à la production.

Avec des modèles plus complexes, plus de données, et des besoins de monitoring, il ne suffit plus que « ça marche dans le notebook ». C’est ici que le **MLOps** entre en jeu. Inspiré du **DevOps**, il structure, automatise et industrialise chaque étape d’un projet ML. 

Pour le démontrer, prenons comme exemple un projet d’analyse de sentiments. L’objectif est de créer un prototype d'IA pour prédire le sentiment des tweets à partir de données open source et de déployer le modèle via une API Cloud.

---

## MLflow : tracer vos expériences

L'une des premières étapes dans un projet ML est le **choix du modèle**. Cette étape essentielle est souvent complexe et dépend des spécificités des données, des contraintes du projet et des objectifs visés. Une approche pragmatique consiste à commencer par un modèle simple, puis à explorer des modèles plus avancés.

Pour notre projet d'analyse de sentiments, nous avons commencé par une **régression logistique**. Pourquoi ? Parce qu’elle est rapide, facile à implémenter et souvent suffisante pour valider une idée. Cette baseline joue un double rôle : 
- Valider rapidement la faisabilité de l'approche.
- Servir de référence pour évaluer les améliorations apportées par des modèles plus complexes.

Pour faciliter la comparaison des expérimentations, nous avons utilisé **MLflow**. Cet outil stocke les métadonnées de chaque run et les modèles associés, tout en garantissant :
- **Traçabilité** des expérimentations.
- **Reproductibilité** des workflows.
- **Gestion centralisée** des versions de modèles.

Dans ce projet, notre modèle de régression logistique a été enregistré dans le **Model Registry** de MLflow comme baseline.
---

## CI/CD et DVC : Structurer et Automatiser vos Pipelines MLOps

Les pipelines de Machine Learning sont bien plus complexes et coûteux que ceux des projets classiques en DevOps. Ils impliquent non seulement du code, mais aussi des données volumineuses, des modèles, et des transformations interconnectées. Ces workflows peuvent durer des heures, nécessitant une gestion rigoureuse pour éviter des calculs inutiles. C’est ici que des outils comme **DVC** et des pratiques de **CI/CD** deviennent essentiels.

### DVC : Structurer les workflows ML
**DVC**, inspiré de Git, permet de structurer les pipelines en :
- Suivant les dépendances entre chaque étape.
- Versionnant les données et les modèles.
- Réexécutant uniquement les étapes impactées par une modification.

Par exemple, dans un projet d’analyse de sentiments, une modification des données entraîne uniquement la réexécution de l’entraînement et de la validation, économisant temps et ressources.

### CI/CD : Automatiser l’intégration et le déploiement

Avec le **CI/CD**, la logique est automatisée :
- **Continuous Integration (CI)** :
  - Validation des données.
  - Entraînement des modèles.
  - Comparaison des performances avec une baseline.
- **Continuous Deployment (CD)** :
  - Déploiement automatique des modèles validés via une API.
  - Surveillance continue des performances en production pour détecter les dégradations.

L’intégration de **DVC** et du **CI/CD** assure des workflows ML reproductibles, industrialisés et adaptables à grande échelle.


## Conclusion

En combinant des outils comme **MLflow**, **DVC** et des pipelines **CI/CD**, le MLOps transforme les expérimentations en projets industrialisés. Cela permet de gérer efficacement les données, les modèles et les étapes d’un pipeline ML tout en réduisant les coûts et en améliorant la reproductibilité.

Ce README offre une introduction aux pratiques MLOps dans le contexte d’un projet d’analyse de sentiments, mais les concepts peuvent être appliqués à tout projet ML nécessitant un passage fiable de l’expérimentation à la production.


