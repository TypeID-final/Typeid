@echo off
echo ==============================================
echo Starting TypeID React Frontend
echo ==============================================
echo.

cd /d "%~dp0"
echo [INFO] Working directory: %CD%
echo [INFO] Running: npm run dev
echo.

start "" "http://localhost:5173"
call npm run dev

echo.
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] React frontend crashed or failed to start.
    echo Please make sure you have run 'npm install' first.
    pause
) else (
    echo [INFO] Frontend stopped normally.
    pause
)
