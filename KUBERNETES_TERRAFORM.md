 # Kubernetes et Terraform sans Azure

Cette configuration cible un cluster Kubernetes local ou on-premise : Minikube, Kind, K3s ou un cluster existant. Terraform gere les ressources de base du namespace et Kubernetes deploye les services applicatifs.

## 1. Demarrer un cluster local

Avec Minikube :

```powershell
minikube start
kubectl config use-context minikube
```

Avec Kind :

```powershell
kind create cluster --name hardtec
kubectl config use-context kind-hardtec
```

## 2. Creer les ressources de base avec Terraform
Depuis PowerShell :

```powershell
cd infra/terraform
Copy-Item terraform.tfvars.example terraform.tfvars
terraform init
terraform fmt -check
terraform validate
terraform plan -out hardtec.tfplan
terraform apply hardtec.tfplan
```

Terraform cree :

- le namespace `hardtec` ;
- la configuration commune des services ;
- le PVC des donnees applicatives ;
- le PVC MLflow.

## 3. Construire l'image localement

Construire une image unique contenant l'API, l'agent et Streamlit :

```powershell
docker build -f Docker/Dockerfile -t hardtec-app:latest .
```

Pour Minikube, rendre l'image disponible dans le cluster :

```powershell
minikube image load hardtec-app:latest
```

Pour Kind :

```powershell
kind load docker-image hardtec-app:latest --name hardtec
```

Le fichier `deploy/kubernetes/kustomization.yaml` utilise deja l'image locale `hardtec-app:latest`.

## 4. Installer l'Ingress Controller

Les manifests utilisent la classe `nginx`. Installer ingress-nginx dans le cluster avant le déploiement, puis configurer DNS ou le fichier hosts pour les domaines de test.

## 5. Creer les secrets

Ne pas appliquer le fichier template tel quel en production. Utiliser le gestionnaire de secrets de votre cluster, ou créer temporairement un secret local :

```powershell
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl -n hardtec create secret generic hardtec-secrets `
  --from-literal=API_SECRET_KEY='<random-secret>' `
  --from-literal=SNOWFLAKE_ACCOUNT='<account>' `
  --from-literal=SNOWFLAKE_USER='<user>' `
  --from-literal=SNOWFLAKE_PASSWORD='<password>'
```

## 6. Deployer l'application

```powershell
kubectl apply -k deploy/kubernetes
kubectl -n hardtec get pods
kubectl -n hardtec get services,ingress
```

## 7. Dashboard AIOps et scalabilité

La page d'accueil Streamlit affiche maintenant le centre de commande AIOps :

- risque courant et niveau de risque ;
- alertes de la fenêtre récente ;
- risque moyen et version du modèle ;
- courbe des risques sur les dernières observations ;
- liste des signaux d'incident nécessitant une revue opérationnelle.

Le dashboard dispose aussi d'un HPA (`autoscaling.yaml`) de 1 à 3 replicas,
avec une stabilisation de montée et de descente. Le serveur Metrics API doit
être installé dans le cluster pour que le HPA fonctionne :

```powershell
kubectl -n hardtec get hpa
kubectl -n hardtec top pods
```

Les PodDisruptionBudgets de `availability.yaml` garantissent qu'un replica API
et dashboard reste disponible pendant une maintenance. L'API conserve un seul
replica par défaut car SQLite n'est pas adapté à une écriture concurrente
multi-replica ; utiliser PostgreSQL avant de scaler l'API horizontalement.

L'authentification API est activée dans Kubernetes (`API_AUTH_ENABLED=true`).
Le dashboard reçoit automatiquement `API_SECRET_KEY` depuis le Secret et le
transmet dans l'en-tête `X-API-Key`.

Le menu Streamlit contient également la page **Support Agent**. En Kubernetes,
elle appelle le service interne `agent:8001` via `AGENT_URL` et affiche la
classification, les tickets historiques similaires, le plan d'actions et la
demande de validation humaine. L'appel à un LLM externe reste facultatif et
nécessite `AGENT_LLM_URL` et `AGENT_LLM_API_KEY`.

## Points importants

- SQLite est conserve ici pour une installation simple avec un seul replica API. Pour plusieurs replicas, utiliser PostgreSQL ou une base de donnees externe.
- Le PVC `standard` permet le partage des donnees selon les capacites du cluster, mais ne remplace pas une base transactionnelle.
- Les domaines `*.hardtec.local` sont des exemples : utiliser un vrai domaine et TLS cert-manager sur un cluster partagé.
- MLflow utilise SQLite dans le PVC pour le bootstrap. Pour une plateforme durable, utiliser PostgreSQL comme backend et un stockage objet compatible S3 pour les artefacts.
- Les secrets ne doivent jamais etre commités dans Git.
