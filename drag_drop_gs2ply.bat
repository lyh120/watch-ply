@echo off
rem Drag one or more 3DGS .ply / .splat files onto THIS file to convert them.
rem Output is written next to each input as <name>_rgb.ply.
setlocal
if "%~1"=="" (
    echo Drag one or more 3DGS .ply / .splat files onto this file.
    pause
    exit /b 1
)
:loop
if "%~1"=="" goto done
python "%~dp0gs2ply.py" --min-opacity 0.1 "%~1"
shift
goto loop
:done
echo.
echo All done. Open the *_rgb.ply files in MeshLab / CloudCompare.
pause
