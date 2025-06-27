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

## Critical Refactoring in Progress

### Current Architecture Challenge

This service currently **duplicates prediction logic** that also exists in `jaqpotpy/api/local_model.py`. This creates:
- **Maintenance burden**: Changes must be made in two places
- **Consistency risks**: Local and production inference may diverge
- **Testing complexity**: Same logic tested in multiple repositories

### Planned Architecture Changes

#### Phase 1: Code Extraction to jaqpotpy (🔄 IN PROGRESS)

The following components will be **moved to jaqpotpy** to create shared prediction logic:

```
🔥 TO BE EXTRACTED TO jaqpotpy/inference/:

src/handlers/
├── predict_sklearn_onnx.py     → jaqpotpy/inference/handlers/sklearn_handler.py
├── predict_torch_onnx.py       → jaqpotpy/inference/handlers/torch_handler.py
├── predict_torch_sequence.py   → jaqpotpy/inference/handlers/torch_sequence_handler.py
└── predict_torch_geometric.py  → jaqpotpy/inference/handlers/torch_geometric_handler.py

src/helpers/
├── predict_methods.py          → jaqpotpy/inference/core/predict_methods.py
├── dataset_utils.py            → jaqpotpy/inference/core/dataset_utils.py
├── model_loader.py             → jaqpotpy/inference/core/model_loader.py
├── recreate_preprocessor.py    → jaqpotpy/inference/core/preprocessor_utils.py
├── image_utils.py              → jaqpotpy/inference/utils/image_utils.py
└── torch_utils.py              → jaqpotpy/inference/utils/tensor_utils.py

src/services/predict_service.py → jaqpotpy/inference/service.py (unified)
```

#### Phase 2: Simplified jaqpotpy-inference (🎯 TARGET STATE)

After refactoring, this repository will be **dramatically simplified**:

```
🎯 SIMPLIFIED STRUCTURE:

src/
├── api/predict.py              # FastAPI endpoint (unchanged)
├── services/
│   └── predict_service.py      # 📝 SIMPLIFIED: Just calls jaqpotpy
├── config/config.py            # Configuration (unchanged)
└── loggers/                    # Logging (unchanged)

main.py                         # FastAPI app (unchanged)
requirements.txt                # 📝 UPDATED: Add jaqpotpy dependency
```

#### New Simplified Prediction Service
```python
# src/services/predict_service.py (POST-REFACTORING)

from jaqpotpy.inference.service import PredictionService
from jaqpot_api_client import PredictionRequest, PredictionResponse

# Global prediction service instance using jaqpotpy shared logic
prediction_service = PredictionService(local_mode=False)

def run_prediction(req: PredictionRequest) -> PredictionResponse:
    """Simplified prediction service using jaqpotpy shared logic."""
    return prediction_service.predict(req)
```

### Benefits of Refactoring

1. **Single Source of Truth**: All prediction logic maintained in jaqpotpy
2. **Guaranteed Consistency**: Local and production inference produce identical results  
3. **Simplified Maintenance**: Changes made once in jaqpotpy, automatically affect production
4. **Enhanced Testing**: Test prediction logic once in jaqpotpy with confidence in production
5. **Reduced Repository Size**: ~70% reduction in code complexity in this repository

### Key Files Affected by Refactoring

#### Files That Will Be Removed/Simplified
- `src/handlers/` → **All handlers moved to jaqpotpy**
- `src/helpers/predict_methods.py` → **Moved to jaqpotpy**
- `src/helpers/dataset_utils.py` → **Moved to jaqpotpy**
- `src/helpers/model_loader.py` → **Enhanced and moved to jaqpotpy**
- `src/helpers/recreate_preprocessor.py` → **Moved to jaqpotpy**
- `src/helpers/image_utils.py` → **Moved to jaqpotpy**
- `src/services/predict_service.py` → **Simplified to 10 lines**

#### Files That Will Remain
- `main.py` → **No changes** (FastAPI app)
- `src/api/predict.py` → **No changes** (endpoint definition)
- `src/config/config.py` → **No changes** (configuration)
- `src/loggers/` → **No changes** (logging infrastructure)
- `src/helpers/s3_client.py` → **Keep** (S3 integration)
- `src/helpers/doa_calc.py` → **Remove** (use jaqpotpy.doa instead)

### Dependencies Changes

#### Current Dependencies (requirements.txt)
```python
# Current: Many ML and processing dependencies
jaqpot-api-client
fastapi
uvicorn
onnxruntime
torch
pandas
numpy
pillow
# ... many others
```

#### Post-Refactoring Dependencies  
```python
# Simplified: Core FastAPI dependencies + jaqpotpy
jaqpotpy>=1.XX.YY      # 🆕 Main dependency with all ML logic
jaqpot-api-client      # Keep for types
fastapi                # Keep for API
uvicorn                # Keep for server
pydantic-settings      # Keep for config
boto3                  # Keep for S3
# Most ML dependencies removed (inherited from jaqpotpy)
```

### Migration Strategy

#### Step 1: Preparation (Current)
- [ ] Test current prediction consistency between jaqpotpy local and jaqpotpy-inference
- [ ] Identify any prediction discrepancies that need fixing
- [ ] Document all handler-specific logic that needs to be preserved

#### Step 2: Extract to jaqpotpy (Next)
- [ ] Create `jaqpotpy/inference/` package structure
- [ ] Move prediction logic from jaqpotpy-inference to jaqpotpy
- [ ] Update jaqpotpy to support both local and production modes
- [ ] Test shared inference service in jaqpotpy

#### Step 3: Simplify jaqpotpy-inference (Final)
- [ ] Update requirements.txt to depend on jaqpotpy
- [ ] Replace prediction logic with jaqpotpy calls
- [ ] Remove duplicate files and dependencies
- [ ] Test simplified production service

### Testing During Refactoring

#### Critical Tests
```bash
# Test current prediction consistency
python test_prediction_consistency.py

# Test specific model types
curl -X POST "http://localhost:8002/predict" -d @sklearn_request.json
curl -X POST "http://localhost:8002/predict" -d @torch_request.json
curl -X POST "http://localhost:8002/predict" -d @geometric_request.json
```

#### Post-Refactoring Validation
```bash
# After refactoring: ensure identical behavior
python test_simplified_inference.py
python test_performance_comparison.py
```

### Rollback Strategy

If issues arise during refactoring:
1. **Feature Flag**: Enable/disable shared logic via environment variable
2. **Backup Branch**: Keep current implementation in separate branch
3. **Gradual Migration**: Migrate model types one at a time
4. **Monitoring**: Monitor prediction accuracy and performance during transition

### Current Status: Pre-Refactoring

⚠️ **Currently using duplicated prediction logic**
⚠️ **Manual synchronization required between jaqpotpy and jaqpotpy-inference**
⚠️ **Changes to prediction algorithms must be made in both repositories**

**Next Action**: Complete jaqpotpy local model testing, then begin Phase 1 refactoring

### Integration Points

#### With jaqpot-api
- Receives prediction requests via REST API
- Handles authentication and authorization through request headers
- Processes model metadata and feature definitions

#### With jaqpotpy (POST-REFACTORING)
- **Primary Dependency**: Will use jaqpotpy for all prediction logic
- **Shared Logic**: Handlers, preprocessing, ONNX inference
- **Consistent Results**: Guaranteed identical predictions to local development

#### Current Integration (PRE-REFACTORING)
- **Independent**: Currently operates independently from jaqpotpy
- **Duplication Risk**: Prediction logic may diverge from jaqpotpy local model
- **Manual Sync**: Requires manual synchronization of changes

This refactoring represents a major architectural shift that will significantly simplify this codebase while improving consistency and maintainability across the Jaqpot ecosystem.