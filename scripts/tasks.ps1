param([ValidateSet('install','lint','test','smoke','demo','api','report')][string]$Task = 'smoke')
$ErrorActionPreference = 'Stop'
$uvPath = if (Get-Command uv -ErrorAction SilentlyContinue) { 'uv' } elseif (Test-Path '.tools/bin/uv.exe') { '.tools/bin/uv.exe' } else { throw 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' }
switch ($Task) {
  'install' { & $uvPath sync --extra dev }
  'lint' {
    & $uvPath run ruff check .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $uvPath run ruff format --check .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $uvPath run mypy src/freightlens
  }
  'test' { & $uvPath run pytest -q }
  'smoke' {
    & $uvPath run freightlens demo --config configs/smoke.yaml
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $uvPath run pytest -q
  }
  'demo' { & $uvPath run freightlens demo --config configs/demo.yaml }
  'api' { & $uvPath run --no-sync uvicorn freightlens.api.app:app --host 127.0.0.1 --port 8000 }
  'report' { & $uvPath run freightlens report --config configs/smoke.yaml }
}
exit $LASTEXITCODE
