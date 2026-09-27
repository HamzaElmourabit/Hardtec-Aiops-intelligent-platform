# Deployment Guide - HARDTEC Intelligent Ticketing Platform

## Overview

This guide covers deploying the HARDTEC AI-Powered IT Incident Detection platform using Docker and Docker Compose.

## Architecture

The system consists of three main services:

```
┌─────────────────────────────────────────────────────────────┐
│                   Docker Network (hardtec_network)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐ ┌─────────────┐ │
│  │  FastAPI Service │  │ Streamlit Service│ │  MinIO      │ │
│  │  (API Backend)   │  │  (Dashboard UI)  │ │ (Storage)   │ │
│  │   Port: 8000     │  │  Port: 8501      │ │ Port: 9000  │ │
│  │                  │  │                  │ │ Console:9001│ │
│  └──────────────────┘  └──────────────────┘ └─────────────┘ │
│          ▲                      ▲                   ▲         │
│          │ /health              │ depends on       │         │
│          │ /predict             │ /history         │         │
│          │ /history             │ /stats           │         │
│          │ /stats               │                  │         │
└──────────┼──────────────────────┼──────────────────┼─────────┘
           │                      │                  │
       Port 8000              Port 8501          Ports 9000/9001
```

## Prerequisites

- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 1.29+ (included with Docker Desktop)
- **Environment Configuration**: Copy `.env.example` to `.env` and configure

## Quick Start

### 1. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your configuration
# For local development, you can use defaults by leaving Snowflake vars as-is
```

### 2. Build and Start Services

```bash
# Build images and start all services
docker-compose up -d

# Or with rebuild (use if you modify code)
docker-compose up -d --build

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f api      # API logs
docker-compose logs -f dashboard # Dashboard logs
docker-compose logs -f minio    # MinIO logs
```

### 3. Verify Deployment

```bash
# Check service health
docker-compose ps

# Test API health endpoint
curl http://localhost:8000/health

# Access dashboard
# Open browser to http://localhost:8501
```

### 4. Stop Services

```bash
# Stop all services (data persists)
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Service Details

### FastAPI Backend (`api` service)

**Purpose**: REST API for ticket classification predictions

**Endpoints**:
- `GET /health` - Service health check
- `POST /predict` - Predict ticket classification
- `GET /history?limit=10` - Retrieve prediction history
- `GET /stats` - Get aggregated statistics

**Environment Variables**:
- `SNOWFLAKE_*` - Snowflake connection credentials (optional)
- `API_HOST` - Bind address (default: 0.0.0.0)
- `API_PORT` - Listen port (default: 8000)

**Volume Mounts**:
- `./src` - Application source code (hot-reload enabled in development)
- `./data` - Data persistence
- `./logs` - Application logs
- `./models` - ML model artifacts

**Health Check**: Enabled, checks every 30s

### Streamlit Dashboard (`dashboard` service)

**Purpose**: Interactive web UI for ticket analysis and predictions

**Features**:
- Real-time ticket prediction form
- Prediction history with timestamps
- Aggregated statistics dashboard
- Priority/Type/Queue distribution charts

**Environment Variables**:
- `SNOWFLAKE_*` - Snowflake connection credentials (optional)
- `STREAMLIT_SERVER_PORT` - Dashboard port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Bind address (default: 0.0.0.0)

**Volume Mounts**:
- `./dashboards` - Dashboard source code (hot-reload)
- `./src` - Application source code
- `./data` - Data persistence
- `./logs` - Application logs

**Dependencies**: Requires API service to be running

### MinIO Object Storage (`minio` service)

**Purpose**: S3-compatible object storage for data lake

**Access Points**:
- **API**: `http://minio:9000`
- **Web Console**: `http://localhost:9001`

**Default Credentials** (from .env):
- Username: `hardtec`
- Password: `hardtec_password`

**Data Persistence**: 
- Volume: `minio_data` (persists across container restarts)

## Production Deployment

### Environment Configuration

For production, create a `.env` file with proper values:

```bash
# Snowflake production credentials
SNOWFLAKE_ACCOUNT=your_prod_account
SNOWFLAKE_USER=your_prod_user
SNOWFLAKE_PASSWORD=your_prod_password
SNOWFLAKE_DATABASE=HARDTEC_PROD
SNOWFLAKE_SCHEMA=PUBLIC

# Security
API_SECRET_KEY=<generate-with-secrets-module>

# Feature flags
ENABLE_SNOWFLAKE_PERSISTENCE=true
ENVIRONMENT=production
DEBUG=false

# API tuning
API_WORKERS=8  # Adjust based on CPU cores
```

### Resource Limits

For production, add resource constraints to `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Reverse Proxy (Nginx Example)

```nginx
upstream api_backend {
    server api:8000;
}

upstream dashboard_ui {
    server dashboard:8501;
}

server {
    listen 80;
    server_name ticketing.example.com;

    # API endpoints
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Dashboard
    location / {
        proxy_pass http://dashboard_ui/;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### SSL/TLS

For production HTTPS, use:
1. **Docker secrets** for certificates
2. **Reverse proxy** (Nginx, Traefik) with SSL termination
3. **Let's Encrypt** for automatic certificate management

## Monitoring & Logging

### Health Checks

Each service includes automated health checks:

```bash
# View health status
docker-compose ps

# Manual health check
curl http://localhost:8000/health
```

### Logs

```bash
# View all logs with timestamps
docker-compose logs -t

# Stream logs from all services
docker-compose logs -f

# View logs from last 100 lines
docker-compose logs --tail=100

# Filter logs
docker-compose logs api | grep ERROR
```

### Performance Tuning

1. **API Workers**: Adjust `API_WORKERS` based on CPU cores
2. **Memory Limits**: Set `--memory` limits per service
3. **Volume Performance**: Use bind mounts on fast storage

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs <service-name>

# Verify Docker daemon
docker ps

# Check resource availability
docker system df
```

### API Connection Errors

```bash
# Check if API is running
curl -v http://localhost:8000/health

# Verify network
docker-compose exec api ping minio

# Check logs
docker-compose logs api | grep -i error
```

### Snowflake Connection Failures

The API gracefully handles Snowflake failures:
- ✅ Predictions stored locally in SQLite
- ⚠️ Warning logged when Snowflake unavailable
- ✅ All functionality continues without Snowflake

### Out of Disk Space

```bash
# Clean up unused Docker resources
docker system prune -a

# View disk usage
docker system df

# Remove old images
docker image prune -a
```

## Data Persistence

### Volumes

- **minio_data**: MinIO object storage (survives container restarts)
- **Bind mounts**: `./data`, `./logs`, `./models` (local filesystem)

### Backup Strategy

```bash
# Backup database
docker-compose exec api cp data/lake/tickets_predictions.db ./backup/

# Backup MinIO
docker-compose exec minio mc mirror minio/data ./backup/minio/

# Backup volumes
docker run --rm -v minio_data:/data -v $(pwd):/backup \
    alpine tar czf /backup/minio_backup.tar.gz -C /data .
```

## Advanced Configuration

### Custom Dockerfile Build

For custom Python packages or system dependencies:

```dockerfile
# Dockerfile customization
RUN apt-get install -y <additional-system-packages>
RUN pip install <additional-python-packages>
```

### Multi-Stage Deployments

Use docker-compose profiles for selective service startup:

```yaml
services:
  api:
    profiles: ["prod", "dev"]
  dashboard:
    profiles: ["prod", "dev"]
  minio:
    profiles: ["dev"]  # Only in development
```

Start specific profile:
```bash
docker-compose --profile dev up
docker-compose --profile prod up
```

### Environment-Specific Overrides

Create additional compose files:

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Security Best Practices

1. **Never commit `.env`** - Use `.env.example` as template
2. **Use Docker secrets** for sensitive credentials in Swarm mode
3. **Enable TLS** for all external connections
4. **Limit network exposure** - Use internal networks only
5. **Regular updates** - Keep base images current
6. **Scan images** - Use `docker scan` for vulnerabilities

## Support

For issues or questions:

1. Check logs: `docker-compose logs -f`
2. Review RAPPORT_ANALYSE_HARDTEC.md for project analysis
3. Consult README.md for general project information
4. Open an issue in version control with logs and steps to reproduce

## Performance Benchmarks

Typical resource usage per service (development):

| Service   | CPU  | Memory | Notes                    |
|-----------|------|--------|--------------------------|
| API       | 0.5% | 150MB  | Varies with request load |
| Dashboard | 0.3% | 200MB  | Streamlit caching        |
| MinIO     | 0.1% | 100MB  | Minimal local setup       |

---

**Last Updated**: 2026-08-31  
**Version**: 1.0.0
