param(
    [string]$InstallDir = "$env:LOCALAPPDATA\NumOCR",
    [switch]$NoDesktopShortcut
)

$ErrorActionPreference = "Stop"

$SourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExeSource = Join-Path $SourceDir "digit_ocr_viewer.exe"
$ModelSource = Join-Path $SourceDir "models"

if (-not (Test-Path -LiteralPath $ExeSource)) {
    throw "digit_ocr_viewer.exe was not found next to install.ps1"
}
if (-not (Test-Path -LiteralPath $ModelSource)) {
    throw "models directory was not found next to install.ps1"
}

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item -LiteralPath $ExeSource -Destination (Join-Path $InstallDir "digit_ocr_viewer.exe") -Force
Copy-Item -LiteralPath $ModelSource -Destination $InstallDir -Recurse -Force

$ReadmeSource = Join-Path $SourceDir "README.md"
if (Test-Path -LiteralPath $ReadmeSource) {
    Copy-Item -LiteralPath $ReadmeSource -Destination (Join-Path $InstallDir "README.md") -Force
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
