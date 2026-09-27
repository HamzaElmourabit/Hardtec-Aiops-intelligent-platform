# 📊 RAPPORT D'ANALYSE: Hardtec Intelligent Ticketing

**Date**: 17 Août 2026  
**Projet**: Système de Classification Automatique de Tickets  
**Évaluation Globale**: ⚠️ **3/10 - NON PRÊT POUR LA PRODUCTION**

---

## 1️⃣ RÉSUMÉ EXÉCUTIF

### Que fait ce projet ?
Ce projet est un **système IA de classification automatique de tickets de support** utilisant 3 modèles ML indépendants pour:
- **Type** (Catégorie: Support Technique, Facturation, etc.)
- **Priorité** (Haute/Moyenne/Basse)
- **Queue** (Routage vers le bon département)

### Architecture Générale
```
Données Brutes → Snowflake → dbt (Bronze/Silver/Gold) → CSV Local → ML Training → API FastAPI
```

### Est-ce un bon projet pour Hardtec ? 🤔
**Réponse**: **NON - Pas dans l'état actuel**

**Points positifs**:
- ✅ Concept solide et utile
- ✅ Architecture dbt bien structurée
- ✅ Pipelines ML organisés
- ✅ Fondations technologiques correctes

**Problèmes critiques**:
- 🔴 **Credentials Snowflake exposées en dur dans le code** ⚠️ CRITIQUE
- 🔴 Aucun test (0% coverage)
- 🔴 Gestion d'erreurs presque inexistante
- 🔴 Zéro documentation
- 🔴 Models circulaires (dépendances croisées)
- 🔴 Pas de monitoring ni versioning

---

## 2️⃣ ANALYSE DÉTAILLÉE PAR DOMAINE

### 🔒 SÉCURITÉ: 2/10 🔴 CRITIQUE

#### ⚠️ Problèmes Critiques:

**1. Credentials Hardcodées**
```python
# src/data/load_to_snowflake.py
password="<your_password>"      # ❌ PASSWORD EXPOSÉ
account="<your_account>"        # ❌ ACCOUNT ID EXPOSÉ
user="<your_username>"          # ❌ USERNAME EXPOSÉ
```

**ACTION IMMÉDIATE REQUISE:**
- [ ] Rotation des credentials Snowflake
- [ ] Supprimer les credentials du code
- [ ] Utiliser variables d'environnement (.env)
- [ ] Ajouter .env à .gitignore
- [ ] Vérifier l'historique Git

**2. Credentials dans les Defaults de l'API**
Les credentials Snowflake peuvent être retrouvées dans les paramètres par défaut.

---

### ⚙️ GESTION D'ERREURS: 3/10 🔴 CRITIQUE

#### Problèmes Identifiés:
```python
# ❌ MAUVAIS: Aucun try/except
def load_data():
    df = pd.read_csv("data/raw/tickets.csv")  # Peut crasher silencieusement
    return df
```

**Absence d'exception handling:**
- ❌ Chargement de données: pas de try/except
- ❌ Connexion Snowflake: pas de timeout ou retry
- ❌ Chargement des modèles: pas de vérification
- ❌ L'API démarre même si les modèles manquent

**Conséquences**:
- Erreurs silencieuses et difficiles à debugger
- L'API retourne des prédictions avec modèles incomplets
- Pas de logs structurés
- Utilise `print()` au lieu de `logging`

---

### 🧪 TESTS: 0/10 🔴 CRITIQUE

**Réalité**: Le dossier `/tests/` est **complètement vide**

**Aucun test de**:
- ❌ Preprocessing des données
- ❌ Entraînement des modèles
- ❌ Endpoint /predict de l'API
- ❌ Performance des modèles
- ❌ Gestion d'erreurs
- ❌ Intégration dbt

**Impact**: Impossible de déployer avec confiance. Chaque changement risque de casser le système.

---

### 📚 DOCUMENTATION: 1/10 🔴 CRITIQUE

**État actuel**:
- ❌ README.md: vide
- ❌ main.py: vide
- ❌ dashboards/streamlit/app.py: vide
- ❌ Aucune docstring dans les fonctions
- ❌ Pas de diagramme architecture
- ❌ Pas de guide d'installation
- ❌ Pas de guide utilisateur
- ❌ Pas de comment contribuer

**Conséquences**: 
- Impossible pour une nouvelle personne de démarrer
- Connaissance concentrée chez une seule personne
- Maintenance difficile

---

### 🧠 QUALITÉ ML: 6/10 🟠 MOYEN

#### Architecture du Modèle

**1. Dépendances Circulaires** ❌
```
Priority Model → besoin de Type + Queue
Queue Model    → besoin de Type + Priority
Type Model     → besoin de Type
```

**Solution actuelle**: Hardcodé `queue="general"` (mauvais !)

**Conséquence**: Le modèle Queue ne peut pas utiliser ses features correctes.

**2. Inconsistance dans les Configurations**
```python
# Type Classifier
LinearSVC(C=1.0)  # ❌ pas de class_weight!

# Priority Classifier
LinearSVC(C=1.0, class_weight="balanced")  # TF-IDF: 30k features

# Queue Classifier
LinearSVC(C=1.5, class_weight="balanced")  # TF-IDF: 40k features ❌ différent!
```

**Problème**: Pas de raison logique pour ces différences. Manque d'optimisation.

**3. Pas de Scores de Confiance**
```python
# API retourne:
{"type": "Technical", "priority": "High", "queue": "L2"}
# Manque:
{"type": "Technical", "priority": "High", "queue": "L2", 
 "confidence": {"type": 0.87, "priority": 0.92, "queue": 0.71}}
```

**Impact**: Impossible de filtrer les prédictions incertaines.

**4. Pas de Tuning des Hyperparamètres**
- ❌ Paramètres LinearSVC hardcodés
- ❌ Pas de GridSearchCV
- ❌ Pas de cross-validation
- ❌ Pas d'étude d'ablation

**5. Déséquilibre de Classes Partiellement Adressé**
- ✅ Utilise `class_weight="balanced"` sur 2/3 modèles
- ❌ Manque SMOTE ou oversampling
- ❌ Pas d'analyse de distribution

**6. Code Dupliqué et Versions Multiples**
```
train_queue_classifier.py      (v1)
train_queue_classifier_v2.py   (v2)
train_queue_classifier_v3.py   (v3) ← utilisé
train_classifier.py            (vide)
```

**Problème**: Plusieurs versions = confusion + maintenance difficile.

---

### 🔗 PIPELINE DE DONNÉES: 6/10 🟠 MOYEN

#### Points Positifs (dbt) ✅
- ✅ Architecture Bronze/Silver/Gold correcte
- ✅ Séparation des transformations
- ✅ Modèles dbt bien nommés

#### Problèmes ❌
- ❌ Pas de tests dbt (`tests/` dbt vide)
- ❌ Pas de freshness checks
- ❌ Pas d'indexes Snowflake
- ❌ Pas de data quality checks

#### Problèmes Preprocessing ❌
```python
# ❌ Préprocessing dans preprocess.py ET dans l'API
# = Risque d'inconsistance entre entraînement et prédiction
```

---

### 📦 DÉPENDANCES & CONFIGURATION: 5/10 🟠 MOYEN

#### Docker ❌
```yaml
# Seulement:
- MinIO (stockage objet)
# Manque:
- API container
- dbt container
- ML training container
- PostgreSQL/autre DB locale
```

**Problème**: Pas complètement containerisé. Difficile à déployer.

#### Fichier requirements.txt ❌
- ⚠️ Encodé en UTF-16 (binaire)
- ❌ Pas de versions lockées
- ❌ Pas de fichier .lock
- ❌ Version Python non spécifiée

#### Gestion de Modèles ❌
- ✅ Utilise joblib (bon choix)
- ❌ Pas de versioning
- ❌ Pas de experiment tracking (MLflow)
- ❌ Pas de model registry

---

### 🏗️ ARCHITECTURE GÉNÉRALE: 4/10 🔴 CRITIQUE

#### Problèmes Architecturaux:

1. **Chargement des Modèles Inefficace**
   - 3 modèles chargés en mémoire au démarrage de l'API
   - Pas de lazy loading
   - Pas de caching

2. **Pas de Monitoring**
   - Aucun tracking des performances
   - Pas de détection de data drift
   - Pas de métriques de prédictions

3. **Pas de Versionning**
   - Impossible de savoir quelle version du modèle est utilisée
   - Pas de rollback

4. **Pas de Fallback**
   - Si un modèle échoue, tout échoue
   - Pas de réponse gracieuse

---

## 3️⃣ SYNTHÈSE DES 25 PROBLÈMES IDENTIFIÉS

| Sévérité | Nombre | Exemples |
|----------|--------|----------|
| 🔴 CRITIQUE | 5 | Credentials exposées, 0 tests, 0 error handling, API sans validation |
| 🟠 HAUTE | 10 | Dépendances circulaires, pas tuning, pas confidence scores |
| 🟡 MOYEN | 10 | Code dupliqué, doc manquante, logging faible |

---

## 4️⃣ RECOMMANDATIONS: PLAN D'ACTION

### 🚨 PHASE 1: SÉCURITÉ (URGENT - Cette semaine)
```
Priority: P0 - Bloquant
Temps: 4-8 heures
```

- [ ] **Rotation Snowflake credentials**
  - Générer nouveaux credentials
  - Tester avec les nouveaux
  - Supprimer les anciens

- [ ] **Externaliser les configurations**
  ```python
  # ❌ Avant:
  password="<your_password>"
  
  # ✅ Après:
  password = os.getenv("SNOWFLAKE_PASSWORD")
  ```

- [ ] **Ajouter .env.template**
  ```
  SNOWFLAKE_USER=<your_user>
  SNOWFLAKE_PASSWORD=<your_password>
  SNOWFLAKE_ACCOUNT=<your_account>
  ```

- [ ] **Configurer .gitignore**
  ```
  .env
  models/
  data/raw/
  data/processed/
  *.joblib
  ```

- [ ] **Audit historique Git**
  - Vérifier les commits avec credentials
  - Utiliser git-filter-repo si nécessaire

---

### 🧪 PHASE 2: TESTS & STABILITÉ (Semaine 1-2)
```
Priority: P1 - Bloquant
Temps: 16-24 heures
```

- [ ] **Ajouter exception handling**
  ```python
  def load_data():
      try:
          df = pd.read_csv("data/raw/tickets.csv")
          return df
      except FileNotFoundError as e:
          logger.error(f"Data file not found: {e}")
          raise
      except pd.errors.ParserError as e:
          logger.error(f"Invalid CSV format: {e}")
          raise
  ```

- [ ] **Implémenter logging structuré**
  ```python
  import logging
  logger = logging.getLogger(__name__)
  logger.info("Model loaded successfully")
  logger.warning("Low confidence prediction")
  logger.error("Model loading failed")
  ```

- [ ] **Validation du modèle au démarrage**
  ```python
  def load_models():
      models = {}
      for model_name in ['type', 'priority', 'queue']:
          path = f'models/{model_name}_classifier.joblib'
          if not os.path.exists(path):
              raise FileNotFoundError(f"Model not found: {path}")
          models[model_name] = joblib.load(path)
      return models
  ```

- [ ] **Écrire tests unitaires**
  ```
  tests/
  ├── test_preprocess.py         # 5-8 tests
  ├── test_api.py               # 8-10 tests
  └── test_models.py            # 5-8 tests
  ```

  Minimum 70% coverage.

---

### 🧠 PHASE 3: AMÉLIORATION ML (Semaine 2-3)
```
Priority: P2 - Important
Temps: 20-30 heures
```

- [ ] **Corriger les dépendances circulaires**
  ```
  Nouvelle architecture:
  1. Type → entrée
  2. Priority → Type
  3. Queue → Type + Priority
  ```

- [ ] **Ajouter confidence scores**
  ```python
  @app.post("/predict")
  def predict(ticket: Ticket):
      predictions = {}
      for model_name, model in models.items():
          pred = model.predict([ticket.text])[0]
          prob = model.decision_function([ticket.text])[0]
          predictions[model_name] = {
              "value": pred,
              "confidence": sigmoid(prob)
          }
      return predictions
  ```

- [ ] **Consolidation du code ML**
  - Supprimer train_v2.py et train_classifier.py
  - Créer `train_pipeline.py` unique
  - Paramètres dans config.yaml

- [ ] **Hyperparameter tuning**
  ```python
  from sklearn.model_selection import GridSearchCV
  
  param_grid = {'C': [0.1, 1, 10], 'class_weight': ['balanced', None]}
  grid = GridSearchCV(LinearSVC(), param_grid, cv=5)
  grid.fit(X_train, y_train)
  ```

- [ ] **Cross-validation**
  ```python
  scores = cross_val_score(model, X, y, cv=5)
  print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
  ```

---

### 📚 PHASE 4: DOCUMENTATION & NETTOYAGE (Semaine 3)
```
Priority: P2 - Important
Temps: 12-16 heures
```

- [ ] **Écrire README.md** incluant:
  - Vue d'ensemble du projet
  - Architecture avec diagramme
  - Instructions d'installation
  - Comment entraîner les modèles
  - Comment utiliser l'API
  - Dépannage

- [ ] **Docstrings dans toutes les fonctions**
  ```python
  def predict_ticket_type(text: str) -> Dict[str, Any]:
      """
      Predict the ticket type using the trained classifier.
      
      Args:
          text: Raw ticket text
          
      Returns:
          Dict with 'type' and 'confidence' keys
          
      Raises:
          ModelNotLoadedError: If model not initialized
      """
  ```

- [ ] **Supprimer code dupliqué**
  - Fusionner train_v*.py
  - Supprimer fichiers vides

- [ ] **Créer diagramme architecture**
  ```
  Raw Data → Snowflake → dbt Bronze/Silver/Gold 
          ↓
      CSV Local → Preprocess → Feature Engineering 
          ↓
      Train Models (Type, Priority, Queue)
          ↓
      Load into API
          ↓
      FastAPI /predict endpoint
          ↓
      Streamlit Dashboard
  ```

---

### 🚀 PHASE 5: MLOps & MONITORING (Semaine 4+)
```
Priority: P3 - Améliorations
Temps: 30-40 heures
```

- [ ] **Model versioning avec MLflow**
  ```python
  import mlflow
  
  with mlflow.start_run():
      mlflow.log_params({"C": 1.0, "kernel": "linear"})
      mlflow.log_metrics({"accuracy": 0.92, "f1": 0.89})
      mlflow.sklearn.log_model(model, "type_classifier")
  ```

- [ ] **Monitoring des prédictions**
  - Dashboard Streamlit avec metrics
  - Alertes si confidence < 70%
  - Tracking du data drift

- [ ] **Pipeline d'entraînement automatisé**
  - Scheduler pour réentraîner mensuellement
  - Auto-validation avant déploiement
  - Rollback automatique si performance↓

- [ ] **Compléter Streamlit dashboard**
  - Pages: Prédictions, Analytics, Monitoring
  - Graphs de performance du modèle
  - Log des erreurs

---

## 5️⃣ PROPOSITIONS DE NOUVEAUX PROJETS/AMÉLIORATIONS

### 💡 Amélioration 1: ANALYTICS DASHBOARD (Immédiat)
**Complexité**: Basse | **Impact**: Haut  
**Temps**: 1-2 jours

Créer un dashboard Streamlit qui affiche:
- Nombre de tickets classifiés par jour
- Distribution des priorités
- Distribution des queues
- Performance des modèles
- Erreurs récentes

**Bénéfice**: Visibilité opérationnelle

---

### 💡 Amélioration 2: FEEDBACK LOOP (Court terme)
**Complexité**: Moyenne | **Impact**: Très haut  
**Temps**: 3-5 jours

Permettre aux opérateurs de corriger les prédictions:
```python
@app.post("/feedback")
def provide_feedback(ticket_id: str, correct_type: str):
    # Enregistrer les erreurs
    # Réentraîner automatiquement chaque 1000 erreurs
```

**Bénéfice**: Amélioration continue du modèle

---

### 💡 Amélioration 3: A/B TESTING (Moyen terme)
**Complexité**: Moyenne | **Impact**: Moyen  
**Temps**: 5-8 jours

Déployer 2 versions du modèle en parallèle:
- 80% trafic: V1 (actuel)
- 20% trafic: V2 (nouveau)

Comparer les performances.

**Bénéfice**: Déploiements plus sûrs

---

### 💡 PROJET NOUVEAU: TICKET SIMILARITY (Pertinent pour Hardtec)
**Complexité**: Haute | **Impact**: Très haut  
**Temps**: 2-3 semaines

Déterminer les tickets similaires/dupliqués:
- Utiliser embeddings (Sentence-BERT)
- Clustering avec HDBSCAN
- Interface pour fusionner les tickets

**Bénéfice**: 
- Réduction des tickets redondants
- Meilleure allocation des ressources
- Économies importantes

---

### 💡 PROJET NOUVEAU: TICKET ROUTING OPTIMIZER
**Complexité**: Très haute | **Impact**: Très haut  
**Temps**: 3-4 semaines

Optimiser le routage basé sur:
- Queue actuelle (charge)
- Expertise de l'opérateur
- Temps de résolution historique
- SLA des clients

**Bénéfice**: 
- Réduction temps de résolution
- Satisfaction client ↑
- Efficacité opérateurs ↑

---

### 💡 PROJET NOUVEAU: TICKET SENTIMENT ANALYSIS
**Complexité**: Moyenne | **Impact**: Moyen  
**Temps**: 1-2 semaines

Analyser sentiment du ticket:
- Positif/Négatif/Neutre
- Urgence émotionnelle
- Prédire satisfaction client

**Bénéfice**: 
- Prioriser tickets insatisfaits
- Escalade automatique
- Meilleure service client

---

## 6️⃣ CHECKLIST POUR HARDTEC

### ✅ Avant Utilisation en Production

- [ ] Sécurité: Credentials externes (Phase 1)
- [ ] Tests: 70% coverage (Phase 2)
- [ ] Error Handling: Complet (Phase 2)
- [ ] Documentation: README + docstrings (Phase 4)
- [ ] Monitoring: Dashboards (Phase 5)
- [ ] Logs: Structurés avec Python logging (Phase 2)

### ✅ Pour Utilisation Opérationnelle

- [ ] Feedback loop: Correction des prédictions
- [ ] Monitoring: Alertes de dégradation
- [ ] Versioning: Traçabilité des modèles
- [ ] Retraining: Pipeline automatisé

---

## 7️⃣ TIMELINE PROPOSÉE

```
Semaine 1: Phases 1-2 (Sécurité + Tests) ← PRIORITÉ
Semaine 2: Phases 3-4 (ML + Doc)
Semaine 3: Phase 5 (MLOps)
Semaine 4+: Nouveaux projets
```

**Estimation totale**: 4-6 semaines pour production-ready

---

## 📋 SCORE FINAL

| Dimension | Score | Verdict |
|-----------|-------|---------|
| Sécurité | 2/10 | 🔴 CRITIQUE |
| Tests | 0/10 | 🔴 CRITIQUE |
| Documentation | 1/10 | 🔴 CRITIQUE |
| Error Handling | 3/10 | 🔴 CRITIQUE |
| ML Quality | 6/10 | 🟠 MOYEN |
| Architecture | 4/10 | 🔴 FAIBLE |
| Data Pipeline | 6/10 | 🟠 MOYEN |
| **GLOBAL** | **3/10** | **❌ NON PRÊT** |

---

## 🎯 CONCLUSION

### Le projet a-t-il du potentiel pour Hardtec ? ✅ OUI

**Pourquoi**:
- Concept utile et pertinent
- Technologie ML moderne
- Données bien structurées (dbt)

### Peut-on le mettre en production maintenant ? ❌ NON

**Pourquoi**:
- Sécurité critique compromise
- Aucune couverture de tests
- Pas de monitoring/alerting
- Documentation inexistante

### Recommendation: ✋ **ARRÊT TEMPORAIRE**

**Action immédiate**:
1. Suivre le plan Phase 1 (Sécurité)
2. Implémenter Phase 2 (Tests)
3. Puis déployer

**Estimé**: 4-6 semaines pour production-ready

---

**Prochaine étape**: Valider ce plan avec l'équipe Hardtec et commencer la Phase 1 dès demain.

