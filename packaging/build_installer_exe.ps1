param(
    [string]$Configuration = "release",
    [string]$OutputExe = "dist\numocr-windows-x64-installer.exe"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputExePath = if ([System.IO.Path]::IsPathRooted($OutputExe)) {
    $OutputExe
} else {
    Join-Path $Root $OutputExe
}
$DistDir = Split-Path -Parent $OutputExePath
$PayloadDir = Join-Path $DistDir "numocr-windows-x64-iexpress"
$SedPath = Join-Path $DistDir "numocr-windows-x64-installer.sed"
$ExePath = Join-Path $Root "target\$Configuration\digit_ocr_viewer.exe"
$ModelDir = Join-Path $Root "models\catia_crnn_digit_ocr_digits"

if (-not (Test-Path -LiteralPath $ExePath)) {
    throw "Release executable not found: $ExePath. Run cargo build --release first."
}
if (-not (Test-Path -LiteralPath (Join-Path $ModelDir "model.onnx"))) {
    throw "ONNX model not found under $ModelDir"
}

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
if (Test-Path -LiteralPath $PayloadDir) {
    $ResolvedPayload = (Resolve-Path -LiteralPath $PayloadDir).Path
    if (-not $ResolvedPayload.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside workspace: $ResolvedPayload"
    }
    Remove-Item -LiteralPath $ResolvedPayload -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $PayloadDir | Out-Null

Copy-Item -LiteralPath $ExePath -Destination (Join-Path $PayloadDir "digit_ocr_viewer.exe") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "install.ps1") -Destination (Join-Path $PayloadDir "install.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "uninstall.ps1") -Destination (Join-Path $PayloadDir "uninstall.ps1") -Force
Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination (Join-Path $PayloadDir "README.md") -Force

foreach ($ModelFile in @("metadata.json", "model.onnx", "model.pt", "onnx_cpu_eval.json")) {
    Copy-Item -LiteralPath (Join-Path $ModelDir $ModelFile) -Destination (Join-Path $PayloadDir $ModelFile) -Force
}

if (Test-Path -LiteralPath $OutputExePath) {
    Remove-Item -LiteralPath $OutputExePath -Force
}

$Sed = @"
[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=NumOCR installation finished.
TargetName=$OutputExePath
FriendlyName=NumOCR Installer
AppLaunched=powershell.exe -ExecutionPolicy Bypass -NoProfile -File install.ps1
PostInstallCmd=<None>
AdminQuietInstCmd=powershell.exe -ExecutionPolicy Bypass -NoProfile -File install.ps1
UserQuietInstCmd=powershell.exe -ExecutionPolicy Bypass -NoProfile -File install.ps1
SourceFiles=SourceFiles

[Strings]
FILE0="digit_ocr_viewer.exe"
FILE1="install.ps1"
FILE2="uninstall.ps1"
FILE3="README.md"
FILE4="metadata.json"
FILE5="model.onnx"
FILE6="model.pt"
FILE7="onnx_cpu_eval.json"

[SourceFiles]
SourceFiles0=$PayloadDir

[SourceFiles0]
%FILE0%=
%FILE1%=
%FILE2%=
%FILE3%=
%FILE4%=
%FILE5%=
%FILE6%=
%FILE7%=
"@

Set-Content -LiteralPath $SedPath -Value $Sed -Encoding ASCII

$Process = Start-Process -FilePath "iexpress.exe" -ArgumentList @("/N", "/Q", $SedPath) -NoNewWindow -Wait -PassThru
if ($Process.ExitCode -ne 0 -and -not (Test-Path -LiteralPath $OutputExePath)) {
    throw "iexpress failed with exit code $($Process.ExitCode)"
}
if (-not (Test-Path -LiteralPath $OutputExePath)) {
    throw "Installer was not created: $OutputExePath"
}

Get-Item -LiteralPath $OutputExePath
exit 0
