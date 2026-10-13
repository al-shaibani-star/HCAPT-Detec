# Regenerate every NSL-KDD table, the learning-curve analysis, the paper numbers and the figures
# from the frozen per-sample predictions (no GPU, no retraining), then compare with the shipped tables.
# Run from the package root:  powershell -ExecutionPolicy Bypass -File revision\reproduce_from_predictions.ps1
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$tables = 'revision\out\tables'
$snap = 'revision\out\tables_shipped'
if (Test-Path $snap) { Remove-Item $snap -Recurse -Force }
Copy-Item $tables $snap -Recurse
$steps = @(
  @('r02_tables_nslkdd.py',            'Tables 7-9, 11 (ablation, per-class recall, Zero-Trust, McNemar)'),
  @('r05_learning_curve_analysis.py',  'Table 10, Figs. 4-5 (learning curve, changeover)'),
  @('r06_calibration_nslkdd.py',       'Tables 12-13, Fig. 7 (calibration and decision rules)'),
  @('r07_validator_independence.py',   'Table 13 lower part, Table A1 (validator dependence)'),
  @('r11_paper_numbers.py',            'paper_numbers.json (every number quoted in the text)'),
  @('r10_figures.py',                  'Figs. 1, 3, 5, 6, 8-10')
)
foreach ($s in $steps) {
  Write-Host "== $($s[1])" -ForegroundColor Cyan
  python "revision\$($s[0])" | Out-Null
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $($s[0])" -ForegroundColor Red; exit 1 }
}
Write-Host "== comparing regenerated tables with the shipped snapshot" -ForegroundColor Cyan
python revision\verify_tables.py $snap $tables
if ($LASTEXITCODE -eq 0) { Write-Host "Reproduction succeeded." -ForegroundColor Green } else { exit 1 }
