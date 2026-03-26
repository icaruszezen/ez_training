@echo off
chcp 65001 >nul
echo.
echo ══════════════════════════════════════════════════════
echo   ez_training 依赖安装脚本
echo   安装 Ultralytics + PyTorch 到 deps\ 目录
echo ══════════════════════════════════════════════════════
echo.

:: ── 检测 Python ────────────────────────────────────────
set PYTHON=
where py >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=py
    goto :found_python
)
where python >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON=python
    goto :found_python
)
echo [错误] 未找到 Python，请先安装 Python 3.10+ 并添加到 PATH
echo        下载地址: https://www.python.org/downloads/
pause
exit /b 1

:found_python
echo 检测到 Python:
%PYTHON% --version
echo.

:: ── 选择 PyTorch 版本 ──────────────────────────────────
:choose_torch
echo 请选择 PyTorch 版本:
echo   1. CUDA 11.8
echo   2. CUDA 12.1  (推荐)
echo   3. CUDA 12.4
echo   4. CPU (无 GPU 加速)
echo.
set /p "choice=请输入数字 [1-4, 默认 2]: "

if "%choice%"=="" set choice=2
if "%choice%"=="1" (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    echo 已选择: CUDA 11.8
    goto :torch_chosen
)
if "%choice%"=="2" (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    echo 已选择: CUDA 12.1
    goto :torch_chosen
)
if "%choice%"=="3" (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    echo 已选择: CUDA 12.4
    goto :torch_chosen
)
if "%choice%"=="4" (
    set TORCH_INDEX=https://download.pytorch.org/whl/cpu
    echo 已选择: CPU
    goto :torch_chosen
)
echo [错误] 无效输入 "%choice%"，请输入 1-4 之间的数字
echo.
goto :choose_torch

:torch_chosen

set "DEPS_DIR=%~dp0deps"
echo.
echo 安装目标: %DEPS_DIR%
echo.

:: ── 安装 ultralytics (阿里云源) ────────────────────────
echo ── 安装 ultralytics ──────────────────────────────────
%PYTHON% -m pip install --target "%DEPS_DIR%" -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com ultralytics
if %errorlevel% neq 0 (
    echo.
    echo [错误] ultralytics 安装失败
    pause
    exit /b 1
)

echo.
:: ── 安装 PyTorch ───────────────────────────────────────
echo ── 安装 PyTorch ──────────────────────────────────────
%PYTHON% -m pip install --target "%DEPS_DIR%" torch torchvision torchaudio --index-url %TORCH_INDEX%
if %errorlevel% neq 0 (
    echo.
    echo [错误] PyTorch 安装失败
    pause
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════
echo   安装完成！请重启 ez_training.exe
echo ══════════════════════════════════════════════════════
pause
