@echo off
setlocal
chcp 65001 >nul
pushd "%~dp0"
set "RAG_PYTHON=.venv\Scripts\python.exe"
if exist "%RAG_PYTHON%" goto install
set "RAG_PYTHON=venv\Scripts\python.exe"
if exist "%RAG_PYTHON%" goto install
py -3.12 -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv .venv
) else (
    python -c "import sys; assert sys.version_info[:2] == (3,12)" >nul 2>&1
    if errorlevel 1 goto missing
    python -m venv .venv
)
if errorlevel 1 goto failed
set "RAG_PYTHON=.venv\Scripts\python.exe"
:install
"%RAG_PYTHON%" -c "import sys; assert sys.version_info[:2] == (3,12)"
if errorlevel 1 goto missing
"%RAG_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto failed
if not exist .env copy /Y .env.example .env >nul
echo.
echo Dependencies installed. Fill your API keys in .env, then double-click the launcher.
goto done
:missing
echo Python 3.12 is required. Install Python 3.12 and run this file again.
goto done
:failed
echo Installation failed. Check network connectivity and the error above.
:done
popd
pause
endlocal
