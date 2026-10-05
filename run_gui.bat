@echo off
setlocal
cd /d "%~dp0"
if defined RESEARCH_OPTIMIZER_PYTHON (
    "%RESEARCH_OPTIMIZER_PYTHON%" -X utf8 "%~dp0app.py"
    goto done
)
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -X utf8 "%~dp0app.py"
    goto done
)
where conda >nul 2>nul
if not errorlevel 1 (
    call conda run --no-capture-output -n base python -X utf8 "%~dp0app.py"
    goto done
)
where python >nul 2>nul
if not errorlevel 1 (
    python -X utf8 "%~dp0app.py"
    goto done
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -X utf8 "%~dp0app.py"
    goto done
)
echo Python was not found. Install Python and the packages in requirements.txt.
pause
exit /b 1
:done
if errorlevel 1 pause
