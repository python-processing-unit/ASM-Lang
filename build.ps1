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
    "$ROOTDIR\asm-lang.py"
Write-Host "Build complete."
Write-Host "Cleaning up build artifacts..."
if (Test-Path "$ROOTDIR\asm-lang.exe") {
    Remove-Item "$ROOTDIR\asm-lang.exe"
}
Remove-Item "$ROOTDIR\build" -Recurse -Force
Move-Item "$ROOTDIR\dist\asm-lang.exe" "$ROOTDIR\"
Remove-Item "$ROOTDIR\dist" -Recurse -Force
Remove-Item "$ROOTDIR\asm-lang.spec"
Write-Host "Cleanup complete. Executable is located at $ROOTDIR\asm-lang.exe"