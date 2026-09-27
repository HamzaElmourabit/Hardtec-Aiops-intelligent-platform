# HARDTEC Intelligent Support & AIOps

## Guide détaillé de génération et de présentation

**Projet :** Plateforme intelligente de support IT, prédiction des incidents et agent IA  
**Formation :** Projet de fin d'année - ENSA Berrechid  
**Période :** 2025-2026

---

## 1. Objectif de la présentation

La présentation doit montrer comment HARDTEC transforme un système de ticketing classique en une plateforme intelligente capable de :

- classifier automatiquement les tickets ;
- prédire leur priorité ;
- les orienter vers la bonne file de traitement ;
- rechercher des cas historiques similaires ;
- assister l'opérateur avec un agent IA ;
- détecter et prédire les risques d'incident ;
- conserver la traçabilité des décisions ;
- être déployée avec Docker et suivie avec MLflow.

La présentation doit rester honnête : le ticketing ML constitue le socle fonctionnel, tandis que certaines briques AIOps restent expérimentales et doivent être présentées avec leurs limites.

---

# Plan des diapositives

## Diapositive 1 - Titre

### Contenu à afficher

**HARDTEC Intelligent Support & AIOps**

Plateforme intelligente de support IT et de prédiction d'incidents.

**Classification | Routage | RAG | Agent IA | AIOps**

Projet de fin d'année - ENSA Berrechid - 2025/2026

### Message oral

Ce projet propose de rapprocher le support utilisateur, l'intelligence artificielle et l'observabilité technique dans une même plateforme. L'objectif est d'aider l'opérateur à prendre une décision plus rapide et mieux contextualisée.

---

## Diapositive 2 - Contexte métier

### Contenu à afficher

- Les tickets IT sont nombreux et hétérogènes.
- Les utilisateurs utilisent des formulations différentes pour un même problème.
- La classification et l'affectation sont souvent manuelles.
- Les historiques de tickets sont difficiles à exploiter.
- Les incidents techniques peuvent être détectés trop tard.

### Message oral

Dans un support classique, l'opérateur doit lire chaque ticket, déterminer son type, estimer son urgence et choisir une équipe. Cette approche devient difficile lorsque le volume augmente et lorsque les tickets sont rédigés dans plusieurs langues ou avec des descriptions peu standardisées.

---

## Diapositive 3 - Problématique et objectifs

### Problématique

> Comment concevoir une plateforme capable de classer automatiquement les tickets, de prédire leur priorité, de proposer une affectation et d'assister l'opérateur grâce aux historiques et aux signaux AIOps ?

### Objectifs

- Automatiser la classification des tickets.
- Prédire la priorité et la file d'affectation.
- Exploiter les tickets historiques avec un moteur RAG.
- Détecter les anomalies et estimer le risque d'incident.
- Générer une recommandation à l'aide d'un agent IA.
- Conserver une validation humaine.
- Assurer la traçabilité avec SQLite et MLflow.
- Préparer un déploiement reproductible avec Docker.

### Message oral

L'objectif n'est pas de remplacer l'opérateur. L'objectif est de lui fournir des informations structurées et contextualisées pour accélérer son travail et réduire les erreurs de tri.

---

## Diapositive 4 - Périmètre réalisé

### Fonctionnalités disponibles

- Pipeline de classification des tickets.
- API REST FastAPI.
- Dashboard Streamlit.
- Persistance SQLite.
- Recherche sémantique avec FAISS.
- Agent IA avec plan d'action.
- Pipeline AIOps et prédiction du risque.
- Artefacts MLflow.
- Scripts Airflow et modèles dbt.
- Configuration Docker.

### Distinction importante

**Fonctionnel :** ticketing, API, historique, dashboard et modèles principaux.  
**Expérimental :** certaines prédictions AIOps, GAIA et MicroSS.  
**Préparatoire :** intégration Power BI et certains flux Snowflake/dbt.

---

## Diapositive 5 - Architecture générale

### Composants

```text
Utilisateur
    |
    v
Streamlit Dashboard
    |
    v
FastAPI
    |
    +--> Classification ML
    +--> RAG / FAISS
    +--> AIOps / risque
    +--> Agent IA
    |
    v
SQLite / DuckDB / fichiers traités
    |
    v
MLflow, Docker, Airflow, dbt
```

### Services documentés

- Streamlit : port 8501.
- FastAPI : port 8000.
- Agent IA : port 8001.
- MLflow : port 5000.
- Airflow : port 8080.

### Message oral

L'architecture sépare l'interface, les services métier, les modèles d'intelligence artificielle et la persistance. Cette séparation facilite les tests et permet de faire évoluer les composants indépendamment.

---

## Diapositive 6 - Flux de traitement d'un ticket

### Flux réel du code

```text
Texte du ticket
    -> Validation API
    -> Prédiction du type
    -> Prédiction de la queue
    -> Prédiction de la priorité
    -> Recommandation
    -> Sauvegarde SQLite
    -> Historique et statistiques
```

### Ordre important

```text
Type -> Queue -> Priority
```

Le type et la queue prédits sont utilisés comme informations supplémentaires pour calculer la priorité.

### Message oral

Le système effectue une prédiction séquentielle. Chaque étape enrichit le contexte transmis à l'étape suivante. Cette organisation permet d'obtenir une décision plus structurée qu'une prédiction indépendante pour chaque champ.

---

## Diapositive 7 - Dataset ticketing

### Chiffres disponibles

- **20 000 tickets** prétraités.
- **4 types :** Incident, Request, Problem, Change.
- **10 queues** de routage.
- **3 priorités :** low, medium, high.
- **2 langues principales :** anglais et allemand.

### Répartition des types

| Type | Nombre |
|---|---:|
| Incident | 7 978 |
| Request | 5 763 |
| Problem | 4 184 |
| Change | 2 075 |

### Répartition des priorités

| Priorité | Nombre |
|---|---:|
| Medium | 8 144 |
| High | 7 801 |
| Low | 4 055 |

### Message oral

La répartition n'est pas parfaitement équilibrée. Il faut donc analyser plusieurs métriques et ne pas se limiter à l'accuracy globale.

---

## Diapositive 8 - Modèles de classification

### Prétraitement

Le fichier `src/data/preprocess.py` nettoie et prépare les tickets avant l'entraînement et l'inférence.

### Modèles principaux

- `models/ticket_type_model.pkl`
- `models/ticket_queue_model_v4.pkl`
- `models/ticket_priority_model_v3.pkl`

### Méthode

- Représentation textuelle TF-IDF.
- Classification supervisée linéaire.
- Sauvegarde des modèles avec Joblib.
- Orchestration dans `src/pipeline/predictor.py`.

### Message oral

Le choix d'un modèle linéaire avec TF-IDF est adapté à un premier système rapide, interprétable et peu coûteux à exécuter. Il constitue une base solide avant une éventuelle évolution vers des modèles transformer spécialisés.

---

## Diapositive 9 - Résultats de classification

### Résultats du classifieur de type

- **Accuracy : 81,98 %**
- **Macro-F1 : 81,80 %**
- **Weighted-F1 : 81,39 %**
- **Support de test : 4 000 tickets**

### F1-score par classe

| Classe | F1-score |
|---|---:|
| Change | 94,42 % |
| Incident | 79,93 % |
| Problem | 54,86 % |
| Request | 97,98 % |

### Interprétation

- Les classes Request et Change sont bien reconnues.
- La classe Problem est plus difficile à distinguer.
- Les modèles de queue et de priorité nécessitent encore des rapports métriques consolidés.

### Message oral

Le résultat global est encourageant, mais la performance de la classe Problem montre qu'une analyse par classe est indispensable. Une bonne accuracy globale ne garantit pas une bonne qualité sur toutes les catégories.

---

## Diapositive 10 - API FastAPI

### Endpoints principaux

| Méthode | Endpoint | Rôle |
|---|---|---|
| GET | `/health` | Vérifier l'état des services |
| POST | `/predict` | Classifier et sauvegarder un ticket |
| POST | `/aiops/predict-risk` | Prédire le risque AIOps |
| POST | `/rag/search` | Rechercher des tickets similaires |
| POST | `/support/analyze` | Analyse ML + RAG + AIOps |
| GET | `/history` | Consulter l'historique |
| GET | `/history/{prediction_id}` | Récupérer une prédiction |
| GET | `/stats` | Statistiques agrégées |

### Validation

- `ticket_text` obligatoire et non vide.
- `top_k` borné entre 1 et 20.
- Gestion des erreurs 400, 404, 500 et 503.
- Validation des requêtes avec Pydantic.

### Message oral

FastAPI sert de point d'entrée central. Le dashboard et les autres clients n'ont pas besoin de connaître directement le fonctionnement interne des modèles.

---

## Diapositive 11 - Dashboard Streamlit

### Pages disponibles

- Dashboard général.
- AI Ticket Analyzer.
- Incident Prediction.
- Intelligent Support.
- Analytics.

### Fonctions

- Saisie d'un ticket.
- Affichage des prédictions.
- Consultation de l'historique.
- Affichage de statistiques.
- Visualisation avec Plotly.
- Appel de l'API FastAPI.

### Message oral

Streamlit fournit une interface adaptée à la démonstration et à l'exploitation interne. L'utilisateur saisit un ticket et obtient rapidement le type, la priorité, la file et la recommandation associée.

---

## Diapositive 12 - Persistance et traçabilité

### SQLite

SQLite stocke :

- le texte du ticket ;
- les prédictions ;
- les scores disponibles ;
- les recommandations ;
- les identifiants et timestamps.

### MLflow

MLflow permet de conserver :

- les expériences ;
- les paramètres ;
- les métriques ;
- les rapports de classification ;
- les artefacts de modèles.

### Message oral

La persistance est importante pour analyser les décisions passées, mesurer l'utilisation du système et améliorer les modèles dans le temps.

---

## Diapositive 13 - RAG et recherche sémantique

### Fonctionnement

```text
Ticket courant
    -> Embedding Sentence Transformer
    -> Recherche FAISS
    -> Tickets historiques similaires
    -> Contexte pour la recommandation
```

### Technologies

- Sentence Transformers.
- Modèle `paraphrase-multilingual-MiniLM-L12-v2`.
- FAISS.
- Documents historiques sérialisés.

### Artefacts

- `models/rag/tickets.index`
- `models/rag/documents.pkl`

### Limite

Le fichier `src/rag/build_index.py` doit encore être complété pour rendre la reconstruction de l'index totalement reproductible.

### Message oral

Le RAG permet de ne pas dépendre uniquement d'une prédiction statistique. Il fournit des exemples historiques qui aident l'opérateur à comprendre pourquoi une recommandation est proposée.

---

## Diapositive 14 - Agent IA

### Fonctionnement

L'agent combine :

1. la prédiction ML ;
2. la recherche de tickets similaires ;
3. les informations de risque AIOps ;
4. la génération d'un plan d'action ;
5. la validation humaine.

### Endpoint

```text
POST /agent/analyze
```

### LLM optionnel

- `AGENT_LLM_URL`
- `AGENT_LLM_API_KEY`

### Principe de sécurité

L'agent propose une action, mais ne déclenche pas directement d'opération irréversible.

### Message oral

L'agent IA est une couche d'orchestration. Il ne remplace pas les modèles spécialisés ; il rassemble leurs résultats pour produire une réponse exploitable par l'opérateur.

---

## Diapositive 15 - AIOps local

### Chaîne de traitement

```text
Signaux temporels
    -> Fenêtres
    -> Anomalies
    -> Incidents
    -> Score de risque
    -> Prédiction
```

### Sorties disponibles

- `data/processed/aiops_windows.csv`
- `data/processed/aiops_anomalies.csv`
- `data/processed/aiops_incidents.csv`
- `data/processed/aiops_incident_risk.csv`
- `data/processed/aiops_incident_predictions.csv`

### Volumes constatés

- 41 fenêtres locales.
- 41 anomalies.
- 41 incidents.
- 37 prédictions d'incident.

### Message oral

Cette partie démontre la chaîne technique de détection et de scoring. Elle doit être présentée comme un prototype AIOps local, car le volume de fenêtres reste limité.

---

## Diapositive 16 - GAIA et MicroSS

### GAIA

- 1 463 fenêtres temporelles.
- 16 502 lignes de logs avec prédictions sémantiques.
- 90 incidents consolidés.
- Évaluation disponible sur 32 incidents.
- Isolation Forest V2 : accuracy 98,15 %, précision 4,02 %, rappel 7,98 %, F1 5,34 %.

### MicroSS

- 17 640 lignes.
- 1 380 features.
- Horizon de prédiction : 15 minutes.
- Précision : 24,50 %.
- Rappel : 69,30 %.
- F1 : 36,20 %.
- ROC-AUC : 0,511.

### Message oral

Les résultats montrent que l'accuracy peut être trompeuse lorsque les incidents sont rares. Pour l'AIOps, il faut surtout suivre la précision, le rappel, le F1-score, le PR-AUC et le nombre de faux positifs.

---

## Diapositive 17 - Industrialisation

### Docker

La configuration active contient principalement :

- API ;
- agent IA ;
- MLflow.

### Airflow

DAGs présents :

- `hardtec_ticket_pipeline.py` ;
- `hardtec_complete_pipeline.py` ;
- `hardtec_aiops_pipeline.py` ;
- `hardtec_rag_pipeline.py`.

### dbt

Architecture des modèles :

```text
Bronze -> Silver -> Gold
```

Modèles principaux :

- `bronze_tickets` ;
- `silver_tickets` ;
- `gold_ticket_analytics` ;
- `gold_ai_predictions`.

### CI/CD

- Installation Python.
- Compilation.
- Tests Pytest.
- Validation Docker Compose.
- Construction et publication GHCR.

### Message oral

Ces outils préparent le passage d'un prototype à une plateforme exploitable. Toutefois, chaque flux doit être validé dans l'environnement cible avant de parler de production complète.

---

## Diapositive 18 - Bilan et perspectives

### Contributions

- Pipeline de ticketing fonctionnel.
- Classification automatique multi-étapes.
- API REST et dashboard.
- Historique et statistiques.
- Recherche sémantique.
- Agent IA supervisé.
- Chaîne AIOps expérimentale.
- Suivi des expériences et déploiement conteneurisé.

### Limites actuelles

- Métriques queue et priorité à compléter.
- Tests à réaligner avec le code actuel.
- Reconstruction du RAG à rendre reproductible.
- Dépendances RAG et MicroSS à expliciter dans les requirements.
- AIOps encore limité par le déséquilibre et le volume des données.
- Aucun déploiement de production complet démontré.

### Perspectives

- Authentification et autorisation de l'API.
- Monitoring et alerting de production.
- Recalibrage continu des modèles.
- Amélioration de la classe Problem.
- Validation des prédictions sur des données réelles.
- Intégration Snowflake et Power BI finalisée.
- Tests end-to-end et tests de charge.

### Conclusion orale

HARDTEC fournit un socle crédible de support intelligent. La partie ticketing est la brique la plus opérationnelle. Les travaux AIOps, GAIA et MicroSS ouvrent des perspectives intéressantes, mais doivent encore être renforcés par davantage de données, de validation et de suivi en production.

---

# Démonstration recommandée

## Scénario 1 - Classification d'un ticket

1. Démarrer l'API FastAPI.
2. Démarrer Streamlit.
3. Saisir un ticket en anglais ou en allemand.
4. Afficher le type prédit.
5. Afficher la queue et la priorité.
6. Montrer la recommandation.
7. Vérifier l'apparition dans l'historique.

## Scénario 2 - Recherche RAG

1. Saisir un ticket technique.
2. Appeler la recherche de tickets similaires.
3. Afficher les documents retournés.
4. Expliquer comment le contexte améliore la recommandation.

## Scénario 3 - Agent IA

1. Envoyer un ticket vers `/agent/analyze`.
2. Afficher le résultat ML.
3. Afficher les tickets similaires.
4. Afficher le plan d'action.
5. Rappeler que la validation humaine reste obligatoire.

## Scénario 4 - AIOps

1. Afficher les fichiers AIOps produits.
2. Montrer les anomalies et le risque calculé.
3. Expliquer la différence entre détection d'anomalie et prédiction d'incident.
4. Présenter les limites des métriques avec transparence.

---

# Commandes utiles

## Générer la présentation PowerPoint

Depuis la racine du projet :

```powershell
.\.venv\Scripts\Activate.ps1
python .\generate_presentation.py
```

Le fichier généré est :

```text
HARDTEC_AIOps_PFA_Presentation.pptx
```

## Vérifier le fichier PowerPoint

```powershell
python -c "from pptx import Presentation; p=Presentation('HARDTEC_AIOps_PFA_Presentation.pptx'); print(len(p.slides), 'slides')"
```

## Lancer l'API

```powershell
python -m uvicorn src.api.main:app --reload --port 8000
```

## Lancer Streamlit

```powershell
streamlit run streamlit/app.py
```

## Lancer MLflow

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

---

# Fichiers de référence

- [generate_presentation.py](generate_presentation.py)
- [rapport_pfa/main.tex](rapport_pfa/main.tex)
- [ARCHITECTURE_COMPLETE.md](ARCHITECTURE_COMPLETE.md)
- [README.md](README.md)
- [MLFLOW.md](MLFLOW.md)
- [src/pipeline/predictor.py](src/pipeline/predictor.py)
- [src/api/main.py](src/api/main.py)
- [streamlit/app.py](streamlit/app.py)
- [data/processed/tickets_clean.csv](data/processed/tickets_clean.csv)
- [reports/mlflow/classification_report.json](reports/mlflow/classification_report.json)
