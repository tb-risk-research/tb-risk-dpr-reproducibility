@echo off
rem =============================================================================
rem tb_risk 一键部署脚本（Windows / PowerShell）
rem
rem 功能：
rem   1. 创建/复用虚拟环境
rem   2. 安装锁定依赖（requirements.lock）
rem   3. 安装项目（-e）
rem   4. 打包离线分发（可选，含模型与配置）
rem   5. 校验依赖与运行自检
rem
rem 用法：
rem   deploy\deploy.bat             标准部署
rem   deploy\deploy.bat --package   部署后打包离线分发
rem =============================================================================
setlocal enabledelayedexpansion

set "ROOT_DIR=%~dp0.."
set "VENV_DIR=%ROOT_DIR%\.venv"
set "PACKAGE=0"

:parse
if "%~1"=="--package" ( set "PACKAGE=1" & shift & goto :parse )
if "%~1"=="" goto :parsed
echo 未知参数: %~1
exit /b 1
:parsed

echo ==^> 项目根目录: %ROOT_DIR%

rem ---- 1. 虚拟环境 ----
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo ==^> 创建虚拟环境...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo 创建虚拟环境失败，请确认已安装 Python 3.9+
    exit /b 1
  )
)
set "PY=%VENV_DIR%\Scripts\python.exe"
echo ==^> Python: 
"%PY%" --version

rem ---- 2. 安装依赖 ----
echo ==^> 安装锁定依赖...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%PY%" -m pip install -r "%ROOT_DIR%\requirements.lock"
if errorlevel 1 (
  echo 依赖安装失败。离线环境请使用预打包 wheel 离线安装。
  exit /b 1
)

rem ---- 3. 安装项目 ----
echo ==^> 安装项目（开发模式）...
"%PY%" -m pip install -e "%ROOT_DIR%"
if errorlevel 1 exit /b 1

rem ---- 4. 可选：打包离线分发 ----
if "%PACKAGE%"=="1" (
  echo ==^> 打包离线分发产物...
  "%PY%" -c "from tb_risk.ops.deploy import pack_distribution; print('打包完成:', pack_distribution())"
)

rem ---- 5. 校验 ----
echo ==^> 依赖状态：
"%PY%" "%ROOT_DIR%\main.py" --check

echo ==^> 部署完成。启动 GUI: tb-risk 或 python main.py
endlocal
