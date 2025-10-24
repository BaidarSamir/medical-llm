# PostgreSQL Setup Script for Medical LLM Project
# This script will create the database and enable pgvector extension

Write-Host "🚀 PostgreSQL Setup for Medical LLM System" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# Set PostgreSQL path
$psqlPath = "C:\Program Files\PostgreSQL\14\bin\psql.exe"

if (-not (Test-Path $psqlPath)) {
    Write-Host "❌ PostgreSQL not found at expected location" -ForegroundColor Red
    Write-Host "Looking for PostgreSQL installation..." -ForegroundColor Yellow
    
    $pgVersions = Get-ChildItem "C:\Program Files\PostgreSQL" -ErrorAction SilentlyContinue
    if ($pgVersions) {
        $version = $pgVersions[0].Name
        $psqlPath = "C:\Program Files\PostgreSQL\$version\bin\psql.exe"
        Write-Host "✅ Found PostgreSQL $version" -ForegroundColor Green
    } else {
        Write-Host "❌ PostgreSQL not installed. Please install it first." -ForegroundColor Red
        exit 1
    }
}

Write-Host "📍 Using PostgreSQL at: $psqlPath" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create database
Write-Host "Step 1: Creating database 'symptom_kb'..." -ForegroundColor Yellow
Write-Host "You will be prompted for the PostgreSQL password (postgres user)" -ForegroundColor Gray
Write-Host ""

$createDbCommand = "CREATE DATABASE symptom_kb;"
& $psqlPath -U postgres -c $createDbCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Database 'symptom_kb' created successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Database might already exist or there was an error." -ForegroundColor Yellow
    Write-Host "If the database already exists, that's fine - continuing..." -ForegroundColor Gray
}

Write-Host ""

# Step 2: Enable pgvector extension
Write-Host "Step 2: Enabling pgvector extension..." -ForegroundColor Yellow
Write-Host "You will be prompted for the password again" -ForegroundColor Gray
Write-Host ""

$enableVectorCommand = "CREATE EXTENSION IF NOT EXISTS vector;"
& $psqlPath -U postgres -d symptom_kb -c $enableVectorCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ pgvector extension enabled successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️  Note: pgvector extension might not be installed." -ForegroundColor Yellow
    Write-Host "The system can still work without it using fallback methods." -ForegroundColor Gray
}

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "✅ PostgreSQL Setup Complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Initialize the knowledge base: py knowledge_base_postgres.py" -ForegroundColor White
Write-Host "2. Start the API server: py simple_api.py" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
