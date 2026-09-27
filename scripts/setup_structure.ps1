$dirs = @(
    "dataset/raw",
    "dataset/processed",
    "dataset/metadata",
    "notebooks/dataset_analysis",
    "notebooks/baseline_model",
    "notebooks/model_training",
    "ml/preprocessing",
    "ml/baseline",
    "ml/resnet_gru",
    "ml/inference",
    "ml/evaluation",
    "computer_vision/detection",
    "computer_vision/tracking",
    "backend/app/api",
    "backend/app/services",
    "backend/app/database",
    "backend/app/models",
    "backend/app/schemas",
    "backend/uploads",
    "backend/snapshots",
    "frontend",
    "docs",
    "tests",
    "scripts"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $keepPath = Join-Path $dir ".gitkeep"
    if (-not (Test-Path $keepPath)) {
        New-Item -ItemType File -Force -Path $keepPath | Out-Null
    }
}
Write-Output "All directories and .gitkeep files verified."
