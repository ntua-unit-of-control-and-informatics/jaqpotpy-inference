# Jaqpot-Inference

`jaqpot-inference` functions as a server for the [jaqpotpy](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy) code, exposing it as an API. This package builds upon the [jaqpotpy-inference](https://github.com/ntua-unit-of-control-and-informatics/jaqpotpy-inference) Docker image (as you can see on the Dockerfile), offering a ready-to-use API server for making predictions.

## Running the Server Locally

To launch the server locally, use the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8002
```
This command initiates the server on port 8002, making the API accessible for local testing and development.

## Docs/Swagger UI

To see the api docs visit http://localhost:8002/docs
