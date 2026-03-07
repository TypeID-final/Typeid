@echo off
echo ==============================================
echo Starting TypeID Flask Backend
echo ==============================================
echo.

cd /d "%~dp0backend"
echo [INFO] Changed directory to backend: %CD%
echo [INFO] Running: python app.py
echo.

python app.py

echo.
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Flask backend crashed or failed to start.
    echo Please check the error message above.
    pause
) else (
    echo [INFO] Backend stopped normally.
    pause
)
