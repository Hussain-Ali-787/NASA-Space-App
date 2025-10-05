param(
  [string]$EnvName = "nasa",
  [string]$Port = "8000"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot

# Ensure conda is available
$conda = (Get-Command conda -ErrorAction SilentlyContinue)
if (-not $conda) {
  Write-Error "Conda not found in PATH. Open Anaconda Prompt or add conda to PATH."
}

# Create/update environment from environment.yml
if (-not (conda env list | Select-String -SimpleMatch " $EnvName ")) {
  conda env create -f backend\environment.yml -n $EnvName
} else {
  conda env update -f backend\environment.yml -n $EnvName
}

# Run server via conda run (no shell activation required)
conda run -n $EnvName python -m uvicorn main:app --host 0.0.0.0 --port $Port --app-dir backend

Pop-Location
