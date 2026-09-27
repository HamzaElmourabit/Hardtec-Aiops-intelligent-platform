#!/bin/bash

# HARDTEC Deployment Manager Script
# Provides convenient commands for managing the Docker-Compose deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default Docker Compose command
COMPOSE_CMD="docker-compose"

# Check if docker-compose is available
if ! command -v $COMPOSE_CMD &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Help text
show_help() {
    cat << EOF
HARDTEC Deployment Manager

Usage: ./deploy.sh [COMMAND] [OPTIONS]

Commands:
    start       Start all services (build if needed)
    stop        Stop all services (keep data)
    restart     Restart all services
    rebuild     Rebuild images and start services
    down        Stop and remove containers (keep volumes)
    clean       Stop, remove containers and volumes (WARNING: removes data)
    
    status      Show status of all services
    logs        Stream logs from all services
    logs-api    Stream logs from API only
    logs-dash   Stream logs from Dashboard only
    logs-minio  Stream logs from MinIO only
    
    health      Check health of all services
    test-api    Test API endpoints
    shell-api   Open shell in API container
    shell-dash  Open shell in Dashboard container
    
    backup      Backup database and volumes
    restore     Restore from backup
    
    help        Show this help message

Examples:
    ./deploy.sh start           # Start all services
    ./deploy.sh logs -f         # Follow logs in real-time
    ./deploy.sh test-api        # Test API health
    ./deploy.sh clean           # Full cleanup (removes data)

Configuration:
    - Copy .env.example to .env before first run
    - Edit .env with your Snowflake credentials
    - Run: ./deploy.sh start

EOF
}

# Check if .env exists
check_env() {
    if [ ! -f "$SCRIPT_DIR/.env" ]; then
        print_warning ".env file not found"
        if [ -f "$SCRIPT_DIR/.env.example" ]; then
            print_info "Creating .env from .env.example..."
            cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
            print_success ".env created. Please update with your credentials."
        else
            print_error "Neither .env nor .env.example found"
            exit 1
        fi
    fi
}

# Start services
cmd_start() {
    print_header "Starting HARDTEC Services"
    check_env
    $COMPOSE_CMD up -d
    print_success "Services started"
    sleep 2
    cmd_status
}

# Stop services
cmd_stop() {
    print_header "Stopping HARDTEC Services"
    $COMPOSE_CMD stop
    print_success "Services stopped"
}

# Restart services
cmd_restart() {
    print_header "Restarting HARDTEC Services"
    cmd_stop
    sleep 1
    cmd_start
}

# Rebuild and start
cmd_rebuild() {
    print_header "Rebuilding Images"
    check_env
    $COMPOSE_CMD up -d --build
    print_success "Images rebuilt and services started"
}

# Bring down services
cmd_down() {
    print_header "Bringing Down Services (keeping volumes)"
    $COMPOSE_CMD down
    print_success "Services stopped"
}

# Clean everything
cmd_clean() {
    print_warning "This will remove all containers and volumes (data will be deleted)"
    read -p "Are you sure? (yes/no) " -r
    echo
    if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        print_header "Cleaning Everything"
        $COMPOSE_CMD down -v
        print_success "Complete cleanup done"
    else
        print_info "Cleanup cancelled"
    fi
}

# Show status
cmd_status() {
    print_header "Service Status"
    $COMPOSE_CMD ps
    echo ""
    print_info "Access points:"
    echo "  - API:       http://localhost:8000"
    echo "  - Dashboard: http://localhost:8501"
    echo "  - MinIO:     http://localhost:9001 (hardtec/hardtec_password)"
}

# Stream logs
cmd_logs() {
    print_header "Streaming Logs"
    $COMPOSE_CMD logs -f "$@"
}

# Health check
cmd_health() {
    print_header "Health Check"
    
    echo -n "API Service: "
    if curl -sf http://localhost:8000/health > /dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy or not responding"
    fi
    
    echo ""
    echo -n "Dashboard Service: "
    if curl -sf http://localhost:8501 > /dev/null 2>&1; then
        print_success "Healthy"
    else
        print_error "Unhealthy or not responding"
    fi
    
    echo ""
    echo -n "MinIO Service: "
    if curl -sf http://localhost:9000/minio/health/live > /dev/null; then
        print_success "Healthy"
    else
        print_error "Unhealthy or not responding"
    fi
}

# Test API
cmd_test_api() {
    print_header "Testing API Endpoints"
    
    BASE_URL="http://localhost:8000"
    
    echo -e "\n${BLUE}1. Health Check${NC}"
    curl -s "$BASE_URL/health" | python -m json.tool || print_error "Health check failed"
    
    echo -e "\n${BLUE}2. Sample Prediction${NC}"
    curl -s -X POST "$BASE_URL/predict" \
        -H "Content-Type: application/json" \
        -d '{"ticket_text": "Network is down", "ticket_type": "Incident"}' | python -m json.tool || print_error "Prediction failed"
    
    echo -e "\n${BLUE}3. History${NC}"
    curl -s "$BASE_URL/history?limit=3" | python -m json.tool || print_error "History fetch failed"
    
    echo -e "\n${BLUE}4. Statistics${NC}"
    curl -s "$BASE_URL/stats" | python -m json.tool || print_error "Stats fetch failed"
    
    print_success "API tests completed"
}

# Shell access
cmd_shell_api() {
    print_info "Opening shell in API container..."
    $COMPOSE_CMD exec api /bin/bash
}

cmd_shell_dash() {
    print_info "Opening shell in Dashboard container..."
    $COMPOSE_CMD exec dashboard /bin/bash
}

# Backup
cmd_backup() {
    print_header "Backing Up Data"
    
    BACKUP_DIR="$SCRIPT_DIR/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    print_info "Backing up database..."
    $COMPOSE_CMD exec api cp data/lake/tickets_predictions.db "$BACKUP_DIR/" 2>/dev/null || print_warning "Database not found"
    
    print_info "Backing up MinIO data..."
    docker run --rm -v minio_data:/data -v "$BACKUP_DIR":/backup \
        alpine tar czf /backup/minio_data.tar.gz -C /data . 2>/dev/null || print_warning "MinIO volume not found"
    
    print_success "Backup completed at: $BACKUP_DIR"
}

# Main command dispatcher
case "${1:-help}" in
    start)      cmd_start ;;
    stop)       cmd_stop ;;
    restart)    cmd_restart ;;
    rebuild)    cmd_rebuild ;;
    down)       cmd_down ;;
    clean)      cmd_clean ;;
    status)     cmd_status ;;
    logs)       shift; cmd_logs "$@" ;;
    logs-api)   cmd_logs api ;;
    logs-dash)  cmd_logs dashboard ;;
    logs-minio) cmd_logs minio ;;
    health)     cmd_health ;;
    test-api)   cmd_test_api ;;
    shell-api)  cmd_shell_api ;;
    shell-dash) cmd_shell_dash ;;
    backup)     cmd_backup ;;
    help|--help|-h) show_help ;;
    *)          print_error "Unknown command: $1"; echo ""; show_help; exit 1 ;;
esac
