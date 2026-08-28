@echo off
REM =========================================================================
REM  VKS3D Installer — pure batch launcher (no hybrid tricks)
REM  Right-click -> Run as administrator
REM =========================================================================
setlocal
net session >nul 2>&1
if errorlevel 1 (
    echo ERROR: Must be run as Administrator.
    echo        Right-click this .bat -^> "Run as administrator".
    pause
    exit /b 1
)

set "VKS3D_DIR=%~dp0"
if "%VKS3D_DIR:~-1%"=="\" set "VKS3D_DIR=%VKS3D_DIR:~0,-1%"

echo.
echo VKS3D Installer: launching PowerShell script...
echo   Scripts dir: %VKS3D_DIR%
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass ^
    -File "%~dp0install.ps1" ^
    -InstallDir "%VKS3D_DIR%"

if errorlevel 1 (
    echo.
    echo INSTALL FAILED (exit code %errorlevel%)
    pause
    exit /b %errorlevel%
)

echo.
echo INSTALL SUCCESS
pause
exit /b 0
