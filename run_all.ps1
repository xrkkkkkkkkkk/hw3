param([switch]$Install)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$localPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
    $Install = $true
}
if ($Install) {
    & $localPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
foreach ($script in @('download_models.py', 'run_experiments.py', 'verify_results.py', 'evaluate.py', 'make_review.py', 'build_report.py')) {
    & $localPython $script
    if ($LASTEXITCODE -ne 0) { throw "Failed while running $script" }
}
Write-Host 'Finished. Open results/listening.html to collect real listening scores.'
