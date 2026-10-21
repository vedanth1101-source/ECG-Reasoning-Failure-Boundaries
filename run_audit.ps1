# ===============================================================================
#  ECG Source Inventory & Disease Prevalence Audit Launcher (PowerShell)
# ===============================================================================
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

python run_audit.py @args
