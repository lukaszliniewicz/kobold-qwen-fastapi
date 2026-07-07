@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PARENT_DIR=%PROJECT_DIR%..\"
for %%I in ("%PARENT_DIR%") do set "PARENT_DIR=%%~fI"

set "PIXI_EXE=%PARENT_DIR%\bin\pixi.exe"
set "ORIGINAL_ARGS=%*"
set "CUSTOM_PIXI=0"

:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="--pixi-path" (
    if "%~2"=="" (
        echo Missing value for --pixi-path.
        exit /b 1
    )
    for %%I in ("%~2") do set "PIXI_EXE=%%~fI"
    set "CUSTOM_PIXI=1"
    shift
    shift
    goto parse_args
)

set "ARG1=%~1"
if /I "!ARG1:~0,12!"=="--pixi-path=" (
    set "PIXI_VALUE=!ARG1:~12!"
    for %%I in ("!PIXI_VALUE!") do set "PIXI_EXE=%%~fI"
    set "CUSTOM_PIXI=1"
    shift
    goto parse_args
)

shift
goto parse_args

:args_done

if "%CUSTOM_PIXI%"=="1" if not exist "%PIXI_EXE%" (
    echo Provided --pixi-path does not exist: "%PIXI_EXE%"
    exit /b 1
)

if "%CUSTOM_PIXI%"=="0" if not exist "%PIXI_EXE%" (
    echo Downloading pixi...
    if not exist "%PARENT_DIR%\bin" mkdir "%PARENT_DIR%\bin"
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%PIXI_EXE%'"
    if errorlevel 1 (
        echo Failed to download pixi.
        exit /b 1
    )
)

set "PIXI_CACHE_DIR=%PARENT_DIR%\.pixi-cache"
set "RATTLER_CACHE_DIR=%PIXI_CACHE_DIR%\rattler"
set "PIP_CACHE_DIR=%PIXI_CACHE_DIR%\pip"
set "UV_CACHE_DIR=%PIXI_CACHE_DIR%\uv-cache"
set "TMP=%PIXI_CACHE_DIR%\tmp"
set "TEMP=%PIXI_CACHE_DIR%\tmp"

if not exist "%PIXI_CACHE_DIR%" mkdir "%PIXI_CACHE_DIR%"
if not exist "%RATTLER_CACHE_DIR%" mkdir "%RATTLER_CACHE_DIR%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%UV_CACHE_DIR%" mkdir "%UV_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

cd /d "%PROJECT_DIR%"
"%PIXI_EXE%" run python run.py %ORIGINAL_ARGS%
