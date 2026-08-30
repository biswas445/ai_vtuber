@echo off
rem ---------------------------------------------------------------------------
rem Starts the AI VTuber companion (overlay window + voice pipeline) with NO
rem visible console window.
rem
rem Double-clicking a .bat always opens a console, so on the first pass this
rem script drops a one-line VBScript into %TEMP% and lets the Windows Script
rem Host relaunch it in a HIDDEN window (style 0); the visible console then
rem closes right away. The hidden second pass does the real work: it builds
rem the renderer and launches the Electron overlay, which spawns the Python
rem voice backend on its own.
rem
rem Quit the app later from the tray icon -> Quit.
rem ---------------------------------------------------------------------------

if /i "%~1"=="hidden" goto run

set "HIDER=%TEMP%\vtuber-start-hidden.vbs"
> "%HIDER%" echo CreateObject("WScript.Shell").Run """%~f0"" hidden", 0, False
wscript //nologo "%HIDER%"
exit /b

:run
del "%TEMP%\vtuber-start-hidden.vbs" 2>nul
cd /d "%~dp0frontend"
call npm start
exit /b
