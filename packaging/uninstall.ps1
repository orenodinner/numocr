param(
    [string]$InstallDir = "$env:LOCALAPPDATA\NumOCR",
    [switch]$KeepInstallDir
)

$ErrorActionPreference = "Stop"

$StartMenuShortcut = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\NumOCR\NumOCR.lnk"
$StartMenuDir = Split-Path -Parent $StartMenuShortcut
$DesktopShortcut = Join-Path ([Environment]::GetFolderPath("Desktop")) "NumOCR.lnk"

Remove-Item -LiteralPath $StartMenuShortcut -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $DesktopShortcut -Force -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $StartMenuDir) {
    $remaining = Get-ChildItem -LiteralPath $StartMenuDir -Force -ErrorAction SilentlyContinue
    if (-not $remaining) {
        Remove-Item -LiteralPath $StartMenuDir -Force
    }
}

if (-not $KeepInstallDir -and (Test-Path -LiteralPath $InstallDir)) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}

Write-Host "NumOCR uninstalled"
