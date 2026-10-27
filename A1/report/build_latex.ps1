param([string]$Compiler)

$ErrorActionPreference = 'Stop'
$reportDir = $PSScriptRoot

if (-not $Compiler) {
    $command = Get-Command tectonic.exe,tectonic,xelatex.exe,xelatex -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
        $Compiler = $command.Source
    } else {
        $localCompiler = Join-Path $env:TEMP 'codex-a1-report-tools\tectonic.exe'
        if (Test-Path -LiteralPath $localCompiler) {
            $Compiler = $localCompiler
        } else {
            throw 'Install Tectonic or XeLaTeX, or supply -Compiler with its full path. See NEXT_STAGES.md.'
        }
    }
}

Push-Location -LiteralPath $reportDir
try {
    if ([System.IO.Path]::GetFileNameWithoutExtension($Compiler) -eq 'tectonic') {
        & $Compiler --keep-logs report.tex
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic compilation failed.' }
    } else {
        for ($pass = 1; $pass -le 2; $pass++) {
            & $Compiler -interaction=nonstopmode -halt-on-error report.tex
            if ($LASTEXITCODE -ne 0) { throw "XeLaTeX pass $pass failed." }
        }
    }
    if (-not (Test-Path -LiteralPath 'report.pdf')) { throw 'No report.pdf was produced.' }
    Write-Output "Compiled: $(Join-Path $reportDir 'report.pdf')"
} finally {
    Pop-Location
}
