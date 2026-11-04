@echo off
REM One-time setup on Windows: virtual environment + dependencies + offline self-test.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python not found. Open "Anaconda Prompt" and run setup.bat from there. & pause & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo pip install failed & pause & exit /b 1)
if not exist .env copy .env.example .env >nul
echo Installing optional PDF export (skipped if it fails)...
pip install markdown playwright && python -m playwright install chromium || echo PDF export not installed - Markdown reports still work.
echo.
echo Running tests...
python -m pytest -q
echo.
echo Setup done. Put your APIFY_TOKEN and TAVILY_API_KEY into .env, then double-click run_all.bat
pause
