@echo off
echo Memulai Aplikasi Klasifikasi Ras Kucing...
echo.
echo Buka browser: http://localhost:8501
echo Tekan Ctrl+C untuk menghentikan
echo.
"C:\Users\surface\AppData\Local\Programs\Python\Python311\Scripts\streamlit.exe" run "%~dp0webapp\app.py" --server.port 8501
pause

