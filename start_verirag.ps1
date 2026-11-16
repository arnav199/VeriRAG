param([ValidateSet('backend', 'frontend')][string]$Service)

$ErrorActionPreference = 'Stop'
$repoRoot = $PSScriptRoot

if ($Service -eq 'backend') {
    Set-Location -LiteralPath $repoRoot
    & (Join-Path $repoRoot '.venv\Scripts\python.exe') -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
    exit $LASTEXITCODE
}

if ($Service -eq 'frontend') {
    Set-Location -LiteralPath (Join-Path $repoRoot 'frontend')
    npm run dev -- --host 0.0.0.0 --port 5173
    exit $LASTEXITCODE
}

$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
$backendScript = $PSCommandPath
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found at $python. Create it and install requirements first."
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot 'frontend\node_modules'))) {
    throw 'Frontend dependencies are missing. Run npm install from the frontend folder first.'
}

foreach ($port in @(8000, 5173)) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        throw "Port $port is already in use (process $($listener.OwningProcess)). Close that app or use its existing server, then run this launcher again."
    }
}

$quotedScript = '"' + $backendScript + '"'
$powershell = Join-Path $PSHOME 'powershell.exe'
Start-Process -FilePath $powershell -ArgumentList "-NoExit -ExecutionPolicy Bypass -File $quotedScript -Service backend" -WorkingDirectory $repoRoot
Start-Process -FilePath $powershell -ArgumentList "-NoExit -ExecutionPolicy Bypass -File $quotedScript -Service frontend" -WorkingDirectory $repoRoot

$backendReady = $false
$frontendReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
        $backendReady = $health.status -eq 'ok'
    } catch { }
    try {
        $page = Invoke-WebRequest -Uri 'http://127.0.0.1:5173/' -TimeoutSec 2 -UseBasicParsing
        $frontendReady = $page.StatusCode -eq 200
    } catch { }
    if ($backendReady -and $frontendReady) { break }
    Start-Sleep -Seconds 1
}

if (-not $backendReady) { Write-Warning 'Backend did not become ready. Check the VeriRAG Backend terminal for errors.' }
if (-not $frontendReady) { Write-Warning 'Frontend did not become ready. Check the VeriRAG Frontend terminal for errors.' }
if ($frontendReady) { Start-Process 'http://localhost:5173' }
if ($backendReady -and $frontendReady) {
    Write-Host 'VeriRAG is ready at http://localhost:5173. Keep both server terminals open.' -ForegroundColor Green
}
