$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonWindow = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonWindow)) {
    $pythonWindow = Join-Path $projectRoot 'venv\Scripts\pythonw.exe'
}
if (-not (Test-Path -LiteralPath $pythonWindow)) {
    throw 'Python environment not found. Run the dependency installer first.'
}
$launcherFile = Join-Path $projectRoot 'launcher.pyw'
Start-Process -FilePath $pythonWindow -ArgumentList ('"' + $launcherFile + '"') -WorkingDirectory $projectRoot -WindowStyle Hidden
