@echo off
REM Full real-data run: check keys -> test one Apify window -> scrape 180 days -> news -> prices -> model -> reports.
cd /d "%~dp0"
if not exist .venv (echo Run setup.bat first. & pause & exit /b 1)
call .venv\Scripts\activate.bat
python run_pipeline.py check || (pause & exit /b 1)
echo.
echo Testing one day of X scraping first...
python run_pipeline.py scrape --max-windows 1 || (pause & exit /b 1)
echo.
echo Scraping the remaining days (resumable - safe to re-run if interrupted)...
python run_pipeline.py all || (pause & exit /b 1)
echo.
echo Done. Open outputs\REPORT.md, outputs\SENTIMENT_OUTLOOK.md and outputs\dashboard.html
start "" outputs\dashboard.html
pause
