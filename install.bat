@echo off
rem ---------------------------------------------------------------------------
rem install.bat - one-shot installer for the "Evil" AI VTuber companion.
rem
rem Sets up everything the project needs on a fresh Windows machine:
rem   1. Python 3.10-3.14 virtual environment in venv\
rem   2. All pinned Python dependencies (torch CUDA stack included)
rem   3. The pocket-tts package itself, installed editable
rem   4. Frontend npm packages + a verified renderer build
rem   5. A pocket_tts\.env configuration template, if none exists yet
rem   6. Optional: pre-downloads every AI model so the first launch is instant
rem
rem Safe to re-run: an existing venv, node_modules and .env are kept.
rem
rem Usage:  install.bat           interactive
rem         install.bat --yes     non-interactive, default answers, no pauses
rem ---------------------------------------------------------------------------

setlocal EnableExtensions
cd /d "%~dp0"

set "VENV_PY=%~dp0venv\Scripts\python.exe"
set "ENV_FILE=%~dp0pocket_tts\.env"
if /i "%~1"=="--yes" set "YES=1"

echo ============================================================
echo   EVIL - AI VTuber companion - installer
echo ============================================================
echo.

rem ----- prerequisite: Python 3.10 - 3.14 ------------------------------------
set "PYEXE="
for %%V in (3.14 3.13 3.12 3.11 3.10) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    python -c "import sys;sys.exit(0 if (3,10)<=sys.version_info[:2]<(3,15) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYEXE=python"
)
if not defined PYEXE (
    echo [error] No Python 3.10 - 3.14 found on PATH or the py launcher.
    echo         Install Python 3.14 from https://www.python.org/downloads/
    echo         and check "Add python.exe to PATH" during setup.
    goto :fail
)
echo [ok] Python found: %PYEXE%

rem ----- prerequisite: Node.js ------------------------------------------------
where node >nul 2>nul
if errorlevel 1 (
    echo [error] Node.js was not found on PATH.
    echo         Install the LTS build from https://nodejs.org/
    goto :fail
)
echo [ok] Node.js found:
node --version
echo.

rem ----- 1/6: virtual environment --------------------------------------------
if exist "%VENV_PY%" (
    echo [1/6] venv\ already exists - reusing it.
) else (
    echo [1/6] creating Python virtual environment in venv\...
    %PYEXE% -m venv venv
    if errorlevel 1 goto :fail
)
if not exist "%VENV_PY%" goto :fail

rem ----- 2/6: Python dependencies --------------------------------------------
echo [2/6] installing pinned Python dependencies...
echo        NOTE: the CUDA torch stack is a large one-time download.
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

rem ----- 3/6: pocket-tts package ---------------------------------------------
echo [3/6] installing the pocket-tts package - editable...
"%VENV_PY%" -m pip install -e .
if errorlevel 1 goto :fail

rem ----- 4/6: frontend --------------------------------------------------------
echo [4/6] installing frontend packages and building the renderer...
pushd "%~dp0frontend"
call npm install
if errorlevel 1 goto :frontend_fail
call npm run build
if errorlevel 1 goto :frontend_fail
popd
goto :frontend_done

:frontend_fail
popd 2>nul
goto :fail

:frontend_done

rem ----- 5/6: configuration ---------------------------------------------------
set "NEEDKEY="
if exist "%ENV_FILE%" (
    echo [5/6] pocket_tts\.env exists - keeping your configuration.
) else (
    echo [5/6] writing a pocket_tts\.env template...
    >  "%ENV_FILE%" echo # Runtime configuration for the Evil voice pipeline.
    >> "%ENV_FILE%" echo # The ONLY required entry is GROQ_API_KEY.
    >> "%ENV_FILE%" echo # Free key: https://console.groq.com/keys
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo GROQ_API_KEY=
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo # Everything below has a built-in default. Remove the
    >> "%ENV_FILE%" echo # leading "# " on a line to override it.
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo # --- LLM - Groq ---------------------------------------------------
    >> "%ENV_FILE%" echo # LLM_MODEL=qwen/qwen3.8-27b
    >> "%ENV_FILE%" echo # MAX_HISTORY_TURNS=12
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo # --- speech-to-text - faster-whisper -------------------------------
    >> "%ENV_FILE%" echo # STT_MODEL=distil-medium.en
    >> "%ENV_FILE%" echo # STT_DEVICE=cuda
    >> "%ENV_FILE%" echo # STT_COMPUTE_TYPE=float32
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo # --- voice activity detection - silero -------------------------------
    >> "%ENV_FILE%" echo # VAD_THRESHOLD=0.5
    >> "%ENV_FILE%" echo # VAD_MIN_SILENCE_MS=600
    >> "%ENV_FILE%" echo # VAD_SPEECH_PAD_MS=300
    >> "%ENV_FILE%" echo # VAD_DEVICE=cpu
    >> "%ENV_FILE%" echo # VAD_DTYPE=float32
    >> "%ENV_FILE%" echo # VAD_MIN_VOICED_FRACTION=0.35
    >> "%ENV_FILE%" echo # VAD_NOISE_MARGIN=3.0
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo # --- text-to-speech - kyutai pocket-tts ------------------------------
    >> "%ENV_FILE%" echo # TTS_VOICE=azelma
    >> "%ENV_FILE%" echo # TTS_DEVICE=cuda
    >> "%ENV_FILE%" echo # TTS_DTYPE=float32
    >> "%ENV_FILE%" echo.
    >> "%ENV_FILE%" echo # --- audio devices - blank means system default ----------------------
    >> "%ENV_FILE%" echo # Match by device-name substring, e.g. INPUT_DEVICE=Yeti
    >> "%ENV_FILE%" echo # INPUT_DEVICE=
    >> "%ENV_FILE%" echo # OUTPUT_DEVICE=
)
findstr /r /c:"^GROQ_API_KEY=[a-zA-Z0-9]" "%ENV_FILE%" >nul 2>nul
if errorlevel 1 set "NEEDKEY=1"

if defined NEEDKEY (
    echo.
    echo [action needed] GROQ_API_KEY is not set in pocket_tts\.env.
    echo   The voice brain cannot start without it.
    if defined YES (
        echo   Open pocket_tts\.env in a text editor and paste your key.
    ) else (
        set /p "OPENENV=Open it in Notepad now to paste your key? [Y/n]: "
    )
)
if defined NEEDKEY if not defined YES if /i not "%OPENENV%"=="n" start "" notepad "%ENV_FILE%"

rem ----- 6/6: model pre-download ---------------------------------------------
echo.
set "PRECACHE="
if defined YES (
    set "PRECACHE=y"
) else (
    set /p "PRECACHE=[6/6] Download every AI model now so the first launch is instant? [Y/n]: "
)
if /i not "%PRECACHE%"=="n" (
    echo       Loading Silero VAD + faster-whisper + pocket-tts models once,
    echo       so they are cached. One-time, several GB on a cold cache.
    "%VENV_PY%" install_precache.py
    if errorlevel 1 echo [warn] model pre-download incomplete - first launch finishes it.
)

echo.
echo ============================================================
echo   Install complete.
echo   Start the companion with:  Hikasha.bat
echo ============================================================
if not defined YES pause
exit /b 0

:fail
echo.
echo [error] Installation failed - see the messages above.
if not defined YES pause
exit /b 1
