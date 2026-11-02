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
where py >nul 2>nul
if not errorlevel 1 set PY=py
echo Instalando as bibliotecas e o PyInstaller...
%PY% -m pip install numpy scipy matplotlib pandas openpyxl pyinstaller
if errorlevel 1 (
  echo.
  echo Nao foi possivel instalar as bibliotecas. Verifique a internet e tente de novo.
  pause
  exit /b 1
)
echo.
echo Gerando o executavel - pode levar alguns minutos...
set ICONE=
if exist "CRAS.ico" set ICONE=--icon CRAS.ico
%PY% -m PyInstaller --noconfirm --onefile --windowed --name CRAS %ICONE% CRAS_desktop.py
echo.
if exist "dist\CRAS.exe" (
  echo Pronto! O programa esta em: %cd%\dist\CRAS.exe
  explorer dist
) else (
  echo Algo deu errado. Veja as mensagens acima.
)
pause
