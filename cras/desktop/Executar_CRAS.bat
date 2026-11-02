@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist "CRAS_desktop.py" (
  echo Nao encontrei o arquivo CRAS_desktop.py nesta pasta:
  echo %cd%
  echo.
  echo Se voce abriu este arquivo de dentro do .zip, extraia o .zip primeiro
  echo - botao direito, Extrair tudo - ou use a pasta Downloads\CRAS_desktop.
  pause
  exit /b 1
)
set PY=python
set PYW=pythonw
where py >nul 2>nul
if not errorlevel 1 (
  set PY=py
  set PYW=pyw
)
%PY% -c "import numpy, scipy, matplotlib, pandas, openpyxl" 2>nul
if errorlevel 1 (
  echo Instalando as bibliotecas necessarias - so acontece na primeira vez...
  %PY% -m pip install numpy scipy matplotlib pandas openpyxl
  if errorlevel 1 (
    echo.
    echo Nao foi possivel instalar as bibliotecas. Verifique se o Python esta instalado e se ha internet.
    pause
    exit /b 1
  )
)
start "" %PYW% CRAS_desktop.py
