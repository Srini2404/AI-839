#!/bin/bash

# Echo off equivalent in bash is to just not use 'set -x' to avoid command printing

# Activate conda environment (uncomment if needed)
echo "Activating conda environment..."
conda activate mlops

# Generate the Dockerfile from the MLflow model
echo "Generating Dockerfile from MLflow model..."
mlflow models generate-dockerfile -m "models:/loan_pred_hw06/latest"
echo "Generated Dockerfile"

# Change directory to where the mlflow-dockerfile is located
cd mlflow-dockerfile || exit

# Run PowerShell-like command to insert lines at 15 and 16 into Dockerfile using bash and sed
echo "Inserting lines into Dockerfile..."
sed -i '15iCOPY /Package /opt/mlflow/Package' Dockerfile
sed -i '16iRUN pip install /opt/mlflow/Package/*' Dockerfile
echo "Inserted lines into Dockerfile"

# Start Docker Desktop if not already running (this is specific to macOS; Linux may differ)
open -a Docker

# Wait for Docker to start
echo "Waiting for Docker Desktop to start..."
sleep 10

# Build the Docker image with the provided image name
echo "Building Docker image with name: $1"
docker build -t "$1" .
echo "Built Docker image"

# Run the Docker container using the provided image name
echo "Running Docker container on port 5001 with image: $1"
docker run -p 5001:8080 "$1"
echo "Ran Docker container"

# Pause equivalent in bash (script will finish after Docker run)
read -p "Press any key to exit..."
