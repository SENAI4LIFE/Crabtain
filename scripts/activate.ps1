# Dot-source from the repository root: . .\scripts\activate.ps1
# Activates .venv and applies simulation/config/offline.env to this session.
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $Root '.venv\Scripts\Activate.ps1')
foreach ($line in Get-Content (Join-Path $Root 'simulation\config\offline.env')) {
    if ($line -match '^\s*([A-Z_][A-Z0-9_]*)=(.*)$') {
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
}
$env:MJRL_BACKEND = 'native'
$env:MJRL_DEVICE  = 'cpu'
Write-Host "Crabtain environment active (offline, CPU)."
