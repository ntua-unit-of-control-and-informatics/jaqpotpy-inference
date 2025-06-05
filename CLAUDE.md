# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

- **Run server**: `python main.py` (starts FastAPI server on port 8002)
- **Install dependencies**: `pip install -r requirements.txt`
- **Update dependencies**: `pip-compile --output-file=requirements.txt requirements.in`
- **Lint code**: `ruff check`
- **Format code**: `ruff format`

## Architecture Overview

This is a FastAPI-based inference service for Jaqpotpy machine learning models. The architecture follows a handler-based pattern for different model types:

### Core Components

- **FastAPI App** (`main.py`): Entry point with health check and middleware
- **Prediction API** (`src/api/predict.py`): Single POST endpoint at `/predict`
- **Prediction Service** (`src/services/predict_service.py`): Router that dispatches to appropriate handlers based on model type
- **Model Handlers** (`src/handlers/`): Type-specific prediction logic for:
  - `predict_sklearn_onnx.py`: Scikit-learn ONNX models
  - `predict_torch_onnx.py`: PyTorch ONNX models  
  - `predict_torch_sequence.py`: PyTorch sequence models
  - `predict_torch_geometric.py`: PyTorch Geometric and TorchScript models

### Supporting Infrastructure

- **Configuration** (`src/config/`): Pydantic settings for S3 bucket configuration
- **Helpers** (`src/helpers/`): Utilities for model loading, preprocessing, dataset handling, S3 operations, and domain of applicability calculations
- **Logging** (`src/loggers/`): Custom logger and middleware for request/response logging

### Model Type Dispatching

The service uses pattern matching on `ModelType` enum to route requests:
- `SKLEARN_ONNX` → sklearn handler
- `TORCH_ONNX` → torch ONNX handler  
- `TORCH_SEQUENCE_ONNX` → torch sequence handler
- `TORCH_GEOMETRIC_ONNX` or `TORCHSCRIPT` → torch geometric handler

### Dependencies

Uses Jaqpotpy ecosystem:
- `jaqpotpy`: Core ML library for model creation
- `jaqpot-api-client`: API client with `PredictionRequest`/`PredictionResponse` models
- ONNX runtime for model inference
- PyTorch ecosystem for deep learning models
- S3 integration via boto3 for model storage

## Configuration

Environment variables managed through `pydantic-settings` (loaded from `.env` file):
- `MODELS_S3_BUCKET_NAME`: S3 bucket containing trained models (required)

Create a `.env` file in the project root with:
```
MODELS_S3_BUCKET_NAME=your-bucket-name
```