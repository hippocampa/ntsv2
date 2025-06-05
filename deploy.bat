@echo off
echo Building site...
hugo
if %errorlevel% neq 0 exit /b %errorlevel%

echo Deploying...
cd public
git add .
git commit -m "Site update: %date% %time%"
git push
cd ..
echo Done!