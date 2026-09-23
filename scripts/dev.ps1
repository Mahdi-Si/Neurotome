# Run the API (:7070, auto-reload) and the UI dev server (:5173) together. Ctrl+C stops both.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$neurotome = @(".venv\Scripts\neurotome.exe", ".venv/bin/neurotome") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $neurotome) { $neurotome = "neurotome" }

$api = Start-Process -FilePath $neurotome -ArgumentList "serve", "--reload" -NoNewWindow -PassThru
try {
    Push-Location frontend
    npm run dev
}
finally {
    Pop-Location
    # --reload spawns a worker process; kill the whole tree.
    if (-not $api.HasExited) { taskkill /T /F /PID $api.Id | Out-Null }
}
