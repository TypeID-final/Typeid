@echo off
echo ==============================================
echo Starting TypeID Complete Application
echo ==============================================
echo.

echo [INFO] Starting Flask Backend...
start "TypeID Backend" cmd /c "call "%~dp0start_backend.bat""

echo [INFO] Starting React Frontend...
start "TypeID Frontend" cmd /c "call "%~dp0start_frontend.bat""

echo.
echo Application started!
echo Two new terminal windows should have opened.
echo Keep both windows open while using the application.
echo.
pause
