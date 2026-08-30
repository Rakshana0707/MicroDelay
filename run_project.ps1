# MicroDelay - Run Project Launcher
# This script launches the FastAPI backend and Vite React frontend concurrently.

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "      Starting MicroDelay System             " -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan

# Check port 8000
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($port8000) {
    Write-Host "[!] Port 8000 is already in use. A backend server might already be running." -ForegroundColor Yellow
} else {
    Write-Host "[*] Starting FastAPI Backend on http://127.0.0.1:8000..." -ForegroundColor Green
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd backend; ..\venv\Scripts\uvicorn main:app --reload --host 127.0.0.1 --port 8000" -WindowStyle Normal
}

# Check port 5173
$port5173 = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue
if ($port5173) {
    Write-Host "[!] Port 5173 is already in use. A frontend server might already be running." -ForegroundColor Yellow
} else {
    Write-Host "[*] Starting Vite React Frontend on http://localhost:5173..." -ForegroundColor Green
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -WindowStyle Normal
}

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "MicroDelay is running!" -ForegroundColor Green
Write-Host "  - Backend API: http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host "  - Frontend UI: http://localhost:5173" -ForegroundColor Gray
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Press any key to close this launcher shell..."
Read-Host
