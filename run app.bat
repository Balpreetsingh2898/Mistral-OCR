@echo off
echo Activating the virtual environment...
call .\venv\Scripts\activate

echo Running the app...
streamlit run app.py
echo App started.

pause
