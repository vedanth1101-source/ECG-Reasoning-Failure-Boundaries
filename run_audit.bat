@echo off
rem ===============================================================================
rem  ECG Source Inventory & Disease Prevalence Audit Launcher
rem ===============================================================================
cd /d "%~dp0"
python run_audit.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Script encountered an error or was interrupted.
    pause
)
