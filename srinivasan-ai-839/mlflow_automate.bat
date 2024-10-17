@echo off

REM Activate conda environment
REM conda activate mlops

REM Generate the Dockerfile from the MLflow model
mlflow models generate-dockerfile -m "models:/loan_pred_hw06/latest"

@echo generated Dockerfile

REM Change directory to where the mlflow-dockerfile is located
cd /d "mlflow-dockerfile"

REM Insert lines at 15 and 16 using a temporary file
setlocal enabledelayedexpansion
set "dockerfile=Dockerfile"
set "tempfile=Dockerfile.tmp"
set "line14=COPY /Package /opt/mlflow/Package"
set "line15=RUN pip install /opt/mlflow/Package/*"

(for /f "tokens=*" %%A in ('type "!dockerfile!"') do (
    set "line=%%A"
    echo !line!>>"!tempfile!"
    set /a count+=1
    if !count! == 9 echo !line14!>>"!tempfile!"
    if !count! == 10 echo !line15!>>"!tempfile!"
))

move /y "!tempfile!" "!dockerfile!"

@echo inserted lines into Dockerfile

REM Start Docker Desktop if not already running
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
timeout /t 10 /nobreak >nul  
REM Wait for Docker to start

@echo Docker Desktop started

REM Build the Docker image with the provided image name
docker build -t "%1" .

@echo built Docker image

REM Run the Docker container using the provided image name
docker run -p 5001:8080 "%1"

@echo ran Docker container

pause