param([string]$Compiler)
$ErrorActionPreference='Stop'
if (-not $Compiler) {
    $cmd=Get-Command tectonic.exe,xelatex.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { $Compiler=$cmd.Source }
    else { $Compiler=Join-Path $env:TEMP 'codex-a1-report-tools\methodology-compiler\tectonic.exe' }
}
if (-not (Test-Path -LiteralPath $Compiler)) { throw 'Supply -Compiler with Tectonic or XeLaTeX executable path.' }
$python=Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
Push-Location -LiteralPath $PSScriptRoot
try {
    & $python -X utf8 generate_methodology.py
    if ($LASTEXITCODE -ne 0) { throw 'Model/evidence checks failed.' }
    if ([IO.Path]::GetFileNameWithoutExtension($Compiler) -eq 'tectonic') {
        & $Compiler --keep-logs --keep-intermediates report.tex
        if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
    } else {
        1..2 | ForEach-Object {
            & $Compiler -interaction=nonstopmode -halt-on-error report.tex
            if ($LASTEXITCODE -ne 0) { throw 'Compilation failed.' }
        }
    }
    & $python -X utf8 build_and_verify.py
    if ($LASTEXITCODE -ne 0) { throw 'PDF verification failed.' }
} finally { Pop-Location }
