@echo off
title Voice AI Assistant - Few-Shot Keyword Spotting Demo
echo ========================================================
echo   VOICE AI ASSISTANT - FEW-SHOT KEYWORD SPOTTING GUI
echo   Mo hinh: PyTorch TC-ResNet8 (Exp 025 - Acc: 95.40%%)
echo ========================================================
cd /d "%~dp0"
"C:\Users\ADMIN\AppData\Local\Programs\Python\Python312\python.exe" demo\app_gui_v2.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [LOI] Co van de xay ra khi khoi chay giao dien.
    pause
)
