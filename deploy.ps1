# HARDTEC Deployment Manager Script for PowerShell
# Provides convenient commands for managing the Docker-Compose deployment

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

# Enable color output
$ErrorActionPreference = "Stop"

# Colors
$Colors = @{
    Success = "Green"
    Error   = "Red"
    Warning = "Yellow"
    Info    = "Cyan"
    Header  = "Blue"
}

# Helper functions
function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $Colors.Success
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $Colors.Error
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $Colors.Warning
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $Colors.Info
}

function Write-Header {
    param([string]$Message)
    Write-Host "================================" -ForegroundColor $Colors.Header
    Write-Host $Message -ForegroundColor $Colors.Header
    Write-Host "================================" -ForegroundColor $Colors.Header
}

function Show-Help {
    $helpText = @"
HARDTEC Deployment Manager for PowerShell

Usage: .\deploy.ps1 [COMMAND] [OPTIONS]

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
    
    backup      Backup database and volumes
    help        Show this help message

Examples:
    .\deploy.ps1 start          # Start all services
    .\deploy.ps1 status         # Check service status
    .\deploy.ps1 test-api       # Test API endpoints
    .\deploy.ps1 clean          # Full cleanup (removes data)

Configuration:
    - Copy .env.example to .env before first run
    - Edit .env with your Snowflake credentials
    - Run: .\deploy.ps1 start

"@
    Write-Host $helpText
}

function Check-Env {
    $envPath = Join-Path $PSScriptRoot ".env"
    $examplePath = Join-Path $PSScriptRoot ".env.example"
    
    if (-not (Test-Path $envPath)) {
        Write-Warning-Custom ".env file not found"
        if (Test-Path $examplePath) {
            Write-Info "Creating .env from .env.example..."
            Copy-Item $examplePath $envPath
            Write-Success ".env created. Please update with your credentials."
        } else {
            Write-Error-Custom "Neither .env nor .env.example found"
            exit 1
        }
    }
}

function Cmd-Start {
    Write-Header "Starting HARDTEC Services"
    Check-Env
    docker-compose up -d
    Write-Success "Services started"
    Start-Sleep -Seconds 2
    Cmd-Status
}

function Cmd-Stop {
    Write-Header "Stopping HARDTEC Services"
    docker-compose stop
    Write-Success "Services stopped"
}

function Cmd-Restart {
    Write-Header "Restarting HARDTEC Services"
    Cmd-Stop
    Start-Sleep -Seconds 1
    Cmd-Start
}

function Cmd-Rebuild {
    Write-Header "Rebuilding Images"
    Check-Env
    docker-compose up -d --build
    Write-Success "Images rebuilt and services started"
}

function Cmd-Down {
    Write-Header "Bringing Down Services (keeping volumes)"
    docker-compose down
    Write-Success "Services stopped"
}

function Cmd-Clean {
    Write-Warning-Custom "This will remove all containers and volumes (data will be deleted)"
    $response = Read-Host "Are you sure? (yes/no)"
    
    if ($response -eq "yes") {
        Write-Header "Cleaning Everything"
        docker-compose down -v
        Write-Success "Complete cleanup done"
    } else {
        Write-Info "Cleanup cancelled"
    }
}

function Cmd-Status {
    Write-Header "Service Status"
    docker-compose ps
    Write-Host ""
    Write-Info "Access points:"
    Write-Host "  - API:       http://localhost:8000" -ForegroundColor $Colors.Info
    Write-Host "  - Dashboard: http://localhost:8501" -ForegroundColor $Colors.Info
    Write-Host "  - MinIO:     http://localhost:9001 (hardtec/hardtec_password)" -ForegroundColor $Colors.Info
}

function Cmd-Logs {
    param([string]$Service = "")
    
    Write-Header "Streaming Logs"
    if ($Service) {
        docker-compose logs -f $Service
    } else {
        docker-compose logs -f
    }
}

function Cmd-Health {
    Write-Header "Health Check"
    
    Write-Host "API Service: " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Success "Healthy"
        } else {
            Write-Error-Custom "Unhealthy"
        }
    } catch {
        Write-Error-Custom "Unhealthy or not responding"
    }
    
    Write-Host ""
    Write-Host "Dashboard Service: " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Success "Healthy"
        } else {
            Write-Error-Custom "Unhealthy"
        }
    } catch {
        Write-Error-Custom "Unhealthy or not responding"
    }
    
    Write-Host ""
    Write-Host "MinIO Service: " -NoNewline
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:9000/minio/health/live" -TimeoutSec 5 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Success "Healthy"
        } else {
            Write-Error-Custom "Unhealthy"
        }
    } catch {
        Write-Error-Custom "Unhealthy or not responding"
    }
}

function Cmd-Test-API {
    Write-Header "Testing API Endpoints"
    
    $baseUrl = "http://localhost:8000"
    
    Write-Host ""
    Write-Host "1. Health Check" -ForegroundColor $Colors.Header
    try {
        $response = Invoke-WebRequest -Uri "$baseUrl/health" -TimeoutSec 5
        $response.Content | ConvertFrom-Json | ConvertTo-Json | Write-Host
    } catch {
        Write-Error-Custom "Health check failed"
    }
    
    Write-Host ""
    Write-Host "2. Sample Prediction" -ForegroundColor $Colors.Header
    try {
        $body = @{
            ticket_text = "Network is down"
            ticket_type = "Incident"
        } | ConvertTo-Json
        
        $response = Invoke-WebRequest -Uri "$baseUrl/predict" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 5
        $response.Content | ConvertFrom-Json | ConvertTo-Json | Write-Host
    } catch {
        Write-Error-Custom "Prediction failed"
    }
    
    Write-Host ""
    Write-Host "3. History" -ForegroundColor $Colors.Header
    try {
        $response = Invoke-WebRequest -Uri "$baseUrl/history?limit=3" -TimeoutSec 5
        $response.Content | ConvertFrom-Json | ConvertTo-Json | Write-Host
    } catch {
        Write-Error-Custom "History fetch failed"
    }
    
    Write-Host ""
    Write-Host "4. Statistics" -ForegroundColor $Colors.Header
    try {
        $response = Invoke-WebRequest -Uri "$baseUrl/stats" -TimeoutSec 5
        $response.Content | ConvertFrom-Json | ConvertTo-Json | Write-Host
    } catch {
        Write-Error-Custom "Stats fetch failed"
    }
    
    Write-Success "API tests completed"
}

function Cmd-Backup {
    Write-Header "Backing Up Data"
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupDir = Join-Path $PSScriptRoot "backups\$timestamp"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    Write-Info "Backing up database..."
    try {
        docker-compose exec -T api powershell -Command "Copy-Item data\lake\tickets_predictions.db '$backupDir\'" -ErrorAction SilentlyContinue
    } catch {
        Write-Warning-Custom "Database not found or copy failed"
    }
    
    Write-Success "Backup completed at: $backupDir"
}

# Main command dispatcher
switch ($Command.ToLower()) {
    "start"     { Cmd-Start }
    "stop"      { Cmd-Stop }
    "restart"   { Cmd-Restart }
    "rebuild"   { Cmd-Rebuild }
    "down"      { Cmd-Down }
    "clean"     { Cmd-Clean }
    "status"    { Cmd-Status }
    "logs"      { Cmd-Logs $Args[0] }
    "logs-api"  { Cmd-Logs "api" }
    "logs-dash" { Cmd-Logs "dashboard" }
    "logs-minio" { Cmd-Logs "minio" }
    "health"    { Cmd-Health }
    "test-api"  { Cmd-Test-API }
    "backup"    { Cmd-Backup }
    "help"      { Show-Help }
    "--help"    { Show-Help }
    "-h"        { Show-Help }
    default     {
        Write-Error-Custom "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}
