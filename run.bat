@echo off
REM Logic x AI study (logicxai)
REM Author: KEYUR SHAILESHKUMAR PATEL
REM ORCID: https://orcid.org/0009-0008-5261-043X
REM Affiliations: 1) MSc Artificial Intelligence student, Department of Computer Science, Brunel University London, UK;
REM               2) Independent researcher
REM Note: Conducted independently; not an official Brunel University London project.
REM License: MIT (see LICENSE)
REM One-command run for Windows.  run.bat  (pilot)   or   run.bat full
cd /d %~dp0
set PROFILE=%1
if "%PROFILE%"=="" set PROFILE=pilot
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .
if not exist .env (copy .env.example .env & echo Created .env - paste your API keys into it, then run again. & exit /b 1)
echo == 1. Unit tests
python -m pytest -q || exit /b 1
echo == 2. Offline pipeline test
python -m logicxai run --profile mock_quick || exit /b 1
echo == 3. Which models answer
python -m logicxai check
echo == 4. Time estimate
python -m logicxai plan --profile %PROFILE%
echo == 5. Study run (%PROFILE%)
python -m logicxai run --profile %PROFILE% --until-done
echo Reports are in runs\<run id>\reports\
