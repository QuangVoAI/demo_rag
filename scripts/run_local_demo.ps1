$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = "python"

# Verify python is available
& $pythonPath -c "import sys; sys.exit(0)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Khong tim thay Python"
}

Set-Location $repoRoot

$env:DJANGO_SECURE_SSL_REDIRECT = "false"
$env:DJANGO_SESSION_COOKIE_SECURE = "false"
$env:DJANGO_CSRF_COOKIE_SECURE = "false"

Write-Host "Starting local demo server at http://127.0.0.1:8000/" -ForegroundColor Cyan
Write-Host "Static files served via --insecure for local testing." -ForegroundColor Cyan

& $pythonPath manage.py runserver 127.0.0.1:8000 --insecure
