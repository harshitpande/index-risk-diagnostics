# run_daily.ps1
# PowerShell runner for NIFTY Risk Diagnostics daily pipeline.
#
# Manual run from VS Code terminal:
#   powershell -ExecutionPolicy Bypass -File .\run_daily.ps1

# Configuration
$ProjectRoot  = $PSScriptRoot
$VenvActivate = Join-Path $ProjectRoot "venv\Scripts\Activate.ps1"
$PythonRunner = Join-Path $ProjectRoot "pipeline\run_daily.py"
$LogsDir      = Join-Path $ProjectRoot "logs"
$RunDate      = Get-Date -Format "yyyy-MM-dd"
$PSLogFile    = Join-Path $LogsDir "ps_run_$RunDate.log"

# Ensure logs directory exists
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

# Logging helper
function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$timestamp | $Level | $Message"
    Write-Host $line
    Add-Content -Path $PSLogFile -Value $line -Encoding UTF8
}

# Header
Write-Log "============================================================"
Write-Log "NIFTY Risk Diagnostics -- PowerShell Runner"
Write-Log "Run date : $RunDate"
Write-Log "Project  : $ProjectRoot"
Write-Log "============================================================"

# Weekend check
$DayOfWeek = (Get-Date).DayOfWeek.ToString()
if ($DayOfWeek -eq "Saturday" -or $DayOfWeek -eq "Sunday") {
    Write-Log "Today is $DayOfWeek -- markets closed. Pipeline skipped." "WARN"
    Write-Log "============================================================"
    exit 0
}
Write-Log "Day of week : $DayOfWeek -- proceeding."

# Check virtual environment exists
if (-not (Test-Path $VenvActivate)) {
    Write-Log "Virtual environment not found at: $VenvActivate" "ERROR"
    Write-Log "Create it with: python -m venv venv" "ERROR"
    exit 1
}

# Check runner script exists
if (-not (Test-Path $PythonRunner)) {
    Write-Log "Pipeline runner not found at: $PythonRunner" "ERROR"
    exit 1
}

# Activate venv
Write-Log "Activating virtual environment..."
try {
    & $VenvActivate
    Write-Log "Virtual environment activated."
} catch {
    Write-Log "Failed to activate venv: $_" "ERROR"
    exit 1
}

# Run pipeline
Write-Log "Starting Python pipeline..."
Write-Log "------------------------------------------------------------"

try {
    python $PythonRunner
    $ExitCode = $LASTEXITCODE

    Write-Log "------------------------------------------------------------"

    if ($ExitCode -eq 0) {
        Write-Log "Pipeline completed successfully. Exit code: $ExitCode"
        Write-Log "Outputs saved to: $(Join-Path $ProjectRoot 'outputs')"
    } else {
        Write-Log "Pipeline exited with code: $ExitCode" "ERROR"
        Write-Log "Check Python log at: $(Join-Path $LogsDir "run_$RunDate.log")" "ERROR"
        exit $ExitCode
    }
} catch {
    Write-Log "Unexpected error running pipeline: $_" "ERROR"
    exit 1
}

# Footer
Write-Log "============================================================"
Write-Log "PS runner complete. PS log: $PSLogFile"
Write-Log "============================================================"

exit 0