# Re-run every experiment of the revised manuscript from raw data (GPU; about 15-20 hours in total).
# Every step is resumable: re-running the script skips finished seeds / configurations / trials.
# Run from the package root:  powershell -ExecutionPolicy Bypass -File revision\run_full_pipeline.ps1
# Raw data must be placed as described in DATA.md. Close memory-hungry applications first.
$env:PYTHONIOENCODING = 'utf-8'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$steps = @(
  @('r01_train_nslkdd.py --seeds 0 1 2 3 4',                 'NSL-KDD: 5 treatments x 5 seeds (~3 h)'),
  @('r03_baselines_nslkdd.py --seeds 0 1 2 3 4',             'NSL-KDD: 9 baselines x 5 seeds (~4 h)'),
  @('r04_learning_curve.py --seeds 0 1 2 3 4',               'NSL-KDD: controlled learning curve (~4 h)'),
  @('r09_adversarial_nslkdd.py',                             'NSL-KDD: adversarial stress test (~10 min)'),
  @('r08_dapt_stages.py',                                    'DAPT 2020: three protocols (~20 min)'),
  @('r15_toniot_telemetry_audit.py',                         'ToN-IoT telemetry audit (~1 min)'),
  @('r14_iot_corpora.py --dataset ton_net --part protocols', 'ToN-IoT network: protocols (~1.5 h)'),
  @('r14_iot_corpora.py --dataset edge --part protocols',    'Edge-IIoTset: protocols (~1 h)'),
  @('r14_iot_corpora.py --dataset ton_net --part augment',   'ToN-IoT network: augmentation (~45 min)'),
  @('r14_iot_corpora.py --dataset edge --part augment',      'Edge-IIoTset: augmentation (~45 min)'),
  @('r16_equal_budget_tuning.py --treatment xgb_cw',         'Tuning: XGBoost + class weighting'),
  @('r16_equal_budget_tuning.py --treatment xgb_smote',      'Tuning: XGBoost + SMOTE'),
  @('r16_equal_budget_tuning.py --treatment xgb_cgan',       'Tuning: XGBoost + CGAN'),
  @('r16_equal_budget_tuning.py --treatment xgb_wgan',       'Tuning: XGBoost + WGAN-GP'),
  @('r16_equal_budget_tuning.py --treatment xgb_hybrid',     'Tuning: proposed hybrid'),
  @('r16_equal_budget_tuning.py --treatment lr',             'Tuning: logistic regression'),
  @('r16_equal_budget_tuning.py --treatment dnn',            'Tuning: DNN'),
  @('r16_equal_budget_tuning.py --final',                    'Tuning: 5-seed final scoring (~1 h)'),
  @('r17_unsw_confirmatory.py --run',                        'UNSW-NB15: pre-registered run (~3-4 h)'),
  @('r17_unsw_confirmatory.py --analyze',                    'UNSW-NB15: hypothesis tests'),
  @('r19_unsw_binary_auc.py',                                'UNSW-NB15: post-hoc AUC (not pre-registered)'),
  @('r02_tables_nslkdd.py',                                  'NSL-KDD tables'),
  @('r05_learning_curve_analysis.py',                        'Learning-curve analysis'),
  @('r06_calibration_nslkdd.py',                             'Calibration and decision rules'),
  @('r07_validator_independence.py',                         'Validator independence'),
  @('r11_paper_numbers.py',                                  'Paper numbers'),
  @('r10_figures.py',                                        'Figures')
)
$ok = $true
foreach ($s in $steps) {
  Write-Host "===== $($s[1])" -ForegroundColor Cyan
  $args_ = $s[0].Split(' ')
  python -u "revision\$($args_[0])" @($args_ | Select-Object -Skip 1)
  if ($LASTEXITCODE -ne 0) {
    Write-Host "retrying: $($s[1])" -ForegroundColor Yellow
    python -u "revision\$($args_[0])" @($args_ | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) { Write-Host "STOPPED at: $($s[1]) - run the script again to resume" -ForegroundColor Red; $ok = $false; break }
  }
}
if ($ok) { Write-Host "===== ALL STEPS COMPLETED =====" -ForegroundColor Green }
