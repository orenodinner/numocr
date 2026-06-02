param(
    [string]$InstallDir = "$env:LOCALAPPDATA\NumOCR",
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"

$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExeSource = Join-Path $SourceDir "digit_ocr_viewer.exe"
$ModelSource = Join-Path $SourceDir "models"
$FlatModelOnnx = Join-Path $SourceDir "model.onnx"

if (-not (Test-Path -LiteralPath $ExeSource)) {
    throw "digit_ocr_viewer.exe was not found next to install.ps1"
}
if (-not (Test-Path -LiteralPath $ModelSource) -and -not (Test-Path -LiteralPath $FlatModelOnnx)) {
    throw "models directory or model.onnx was not found next to install.ps1"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -LiteralPath $ExeSource -Destination (Join-Path $InstallDir "digit_ocr_viewer.exe") -Force

if (Test-Path -LiteralPath $ModelSource) {
    Copy-Item -LiteralPath $ModelSource -Destination $InstallDir -Recurse -Force
} else {
    $ModelTarget = Join-Path $InstallDir "models\catia_crnn_digit_ocr_digits"
    New-Item -ItemType Directory -Force -Path $ModelTarget | Out-Null
    foreach ($ModelFile in @("metadata.json", "model.onnx", "model.pt", "onnx_cpu_eval.json")) {
        $SourceFile = Join-Path $SourceDir $ModelFile
        if (Test-Path -LiteralPath $SourceFile) {
            Copy-Item -LiteralPath $SourceFile -Destination (Join-Path $ModelTarget $ModelFile) -Force
        }
    }
}

$ReadmeSource = Join-Path $SourceDir "README.md"
if (Test-Path -LiteralPath $ReadmeSource) {
    Copy-Item -LiteralPath $ReadmeSource -Destination (Join-Path $InstallDir "README.md") -Force
}

$UninstallSource = Join-Path $SourceDir "uninstall.ps1"
if (Test-Path -LiteralPath $UninstallSource) {
    Copy-Item -LiteralPath $UninstallSource -Destination (Join-Path $InstallDir "uninstall.ps1") -Force
}

$WshShell = New-Object -ComObject WScript.Shell
$ExeTarget = Join-Path $InstallDir "digit_ocr_viewer.exe"

$StartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\NumOCR"
New-Item -ItemType Directory -Force -Path $StartMenuDir | Out-Null
$StartMenuShortcut = $WshShell.CreateShortcut((Join-Path $StartMenuDir "NumOCR.lnk"))
$StartMenuShortcut.TargetPath = $ExeTarget
$StartMenuShortcut.WorkingDirectory = $InstallDir
$StartMenuShortcut.Save()

if (-not $NoDesktopShortcut) {
    $DesktopShortcut = $WshShell.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "NumOCR.lnk"))
    $DesktopShortcut.TargetPath = $ExeTarget
    $DesktopShortcut.WorkingDirectory = $InstallDir
    $DesktopShortcut.Save()
}

Write-Host "NumOCR installed to $InstallDir"
