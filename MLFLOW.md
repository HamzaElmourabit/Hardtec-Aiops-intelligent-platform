# MLflow dans HARDTEC

Les scripts d'entrainement des modeles metier enregistrent leurs executions dans l'experience `hardtec-ticketing` :

- `src/ml/train_type_classifier.py`
- `src/ml/train_priority_classifier_v3.py`
- `src/ml/train_queue_classifier_v5.py`

Chaque run contient le hash et la taille du dataset, les hyperparametres, les metriques de classification, le rapport detaille et le fichier du modele genere.

## Utilisation locale

Depuis la racine du projet :

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
python src/ml/train_type_classifier.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Ouvrir ensuite `http://127.0.0.1:5000`.

Pour comparer les quatre variantes du classifieur de files :

```powershell
python src/ml/train_queue_classifier_v5.py
```

Le benchmark V5 cree un run parent et un run enfant par variante (`V5-A` a `V5-D`).

## Serveur MLflow distant

Le tracker lit la variable `MLFLOW_TRACKING_URI`. Exemple :

```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME = "hardtec-ticketing"
```

En production, utiliser un backend SQL et un stockage d'artefacts MinIO plutot que le stockage fichier local.
