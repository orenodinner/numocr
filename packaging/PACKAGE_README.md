# NumOCR Windows Installer

The GitHub release ships `numocr-windows-x64-installer.exe`. Run it directly:

```powershell
.\numocr-windows-x64-installer.exe
```

The installer copies NumOCR to `%LOCALAPPDATA%\NumOCR` and creates Start Menu and desktop shortcuts.

To uninstall:

```powershell
powershell -ExecutionPolicy Bypass -File .\uninstall.ps1
```
