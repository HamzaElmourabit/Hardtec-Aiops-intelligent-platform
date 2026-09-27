# HARDTEC Docker

Cette composition ne depend pas de MinIO. Elle lance l'API FastAPI sur le port 8000,
l'agent IA sur le port 8001 et MLflow sur le port 5000.

```powershell
docker compose -f Docker/docker-compose.yml up --build
```

Test de l'agent :

```powershell
Invoke-RestMethod -Uri http://localhost:8001/agent/analyze -Method Post -ContentType 'application/json' -Body '{"ticket_text":"Le VPN est indisponible pour plusieurs utilisateurs"}'
```# HARDTEC Docker

Cette composition ne depend pas de MinIO. Elle lance :

- FastAPI sur `http://localhost:8000` ;
- l'agent IA sur `http://localhost:8001` ;
- MLflow sur `http://localhost:5000`.

```powershell
docker compose -f Docker/docker-compose.yml up --build
```

Tester l'agent :

```powershell
Invoke-RestMethod `
  -Uri http://localhost:8001/agent/analyze `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"ticket_text":"Le VPN est indisponible pour plusieurs utilisateurs"}'
```