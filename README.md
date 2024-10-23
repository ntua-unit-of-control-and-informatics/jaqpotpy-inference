# jaqpotpy-inference

`jaqpotpy-inference` is a FastAPI application for running models created with [jaqpotpy](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy) and returning predictions. It provides a simple API with documentation generated by Swagger, accessible at the `/docs` endpoint.

## Features

- **FastAPI**: Utilizes the FastAPI framework for high performance and easy API creation.
- **Model Inference**: Runs all models created with Jaqpotpy and returns predictions.
- **API Documentation**: Generates interactive API documentation with Swagger, available at `/docs`.

## Requirements

- Python 3.7+
- FastAPI
- Uvicorn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy-inference.git
   cd jaqpotpy-inference
   ```
2.	Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Running the server

To start the FastAPI server, simply run:
```bash
python main.py
```

The server will be available at http://127.0.0.1:8002.

## API Documentation

Once the server is running, you can access the API documentation at:
http://127.0.0.1:8002/docs

This documentation is auto-generated by Swagger and provides an interactive interface for testing the API endpoints.
