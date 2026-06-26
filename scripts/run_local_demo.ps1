$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repoRoot ".venv\bin\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Khong tim thay Python venv tai $pythonPath"
}

Set-Location $repoRoot

$env:DJANGO_SECURE_SSL_REDIRECT = "false"
$env:DJANGO_SESSION_COOKIE_SECURE = "false"
$env:DJANGO_CSRF_COOKIE_SECURE = "false"

Write-Host "Starting local demo server at http://127.0.0.1:8000/" -ForegroundColor Cyan
Write-Host "Static files served via --insecure for local testing." -ForegroundColor Cyan

& $pythonPath manage.py runserver 127.0.0.1:8000 --insecure
