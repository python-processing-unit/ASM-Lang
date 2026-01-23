$ErrorActionPreference = "Stop"

$ROOTDIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ROOTDIR
Write-Host "Building executable from $ROOTDIR..."
pyinstaller --onefile --icon `
    "$ROOTDIR\docs\icon.png" `
    --add-data "$ROOTDIR\lexer.py;." `
    --add-data "$ROOTDIR\parser.py;." `
    --add-data "$ROOTDIR\interpreter.py;." `
    --add-data "$ROOTDIR\extensions.py;." `
    "$ROOTDIR\prefix.py"
Write-Host "Build complete."
Write-Host "Cleaning up build artifacts..."
if (Test-Path "$ROOTDIR\prefix.exe") {
    Remove-Item "$ROOTDIR\prefix.exe"
}
Remove-Item "$ROOTDIR\build" -Recurse -Force
Move-Item "$ROOTDIR\dist\prefix.exe" "$ROOTDIR\"
Remove-Item "$ROOTDIR\dist" -Recurse -Force
Remove-Item "$ROOTDIR\prefix.spec"
Write-Host "Cleanup complete. Executable is located at $ROOTDIR\prefix.exe"