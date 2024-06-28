import uvicorn
from fastapi import FastAPI
from src.handlers.predict import model_post_handler
from src.entities.prediction_request import PredictionRequestPydantic
from fastapi.responses import JSONResponse

from fastapi import Request
from src.entities import PredictResponseTorch
from src.handlers.prediction_handler_torch import handle_prediction_torch

app = FastAPI()



################################################################################
############################ ONLY FOR LOCAL TESTING ############################
#################################### START #####################################
import sys
# TODO: Replace with local path to the jaqpotpy library for local testing
jaqpotpy_path = "../../PINK-jaqpotpy/jaqpotpy"
running_locally = True
if jaqpotpy_path not in sys.path and running_locally:
    sys.path.append(jaqpotpy_path)
##################################### END ######################################
################################################################################
################################################################################
import jaqpotpy



@app.get('/')
def hello_world():
    return {'Hello': 'World'}

@app.post('/predict/')
def predict(req: PredictionRequestPydantic):
    return JSONResponse(content = model_post_handler(req))

# ################################################################################
# ############################ ONLY FOR LOCAL TESTING ############################
# #################################### START #####################################
# db = {}

# @app.get("/db/")
# def get_db():
#     return {'db': db}


# @app.post("/upload/", 
#              summary="Torch Model Upload",
#              description="Endpoint to upload a PyTorch Model and store it a dictionary (for testing purposes until there is a proper db).")
# async def model_upload(req: Request):
#     """
#     Endpoint to upload a PyTorch model locally using a dictionary as database.
#     """

#     model_data = await req.json()

#     model_data['id'] = len(db.keys())
#     model_id = model_data['id']
#     db[model_id] = model_data

#     return {'model_id': model_id, 'message': "Model uploaded successfully"}

##################################### END ######################################
################################################################################
################################################################################



@app.post("/predict_torch/",
          response_model=PredictResponseTorch,
          summary="Make predictions using a PyTorch model",
          description="Endpoint to make predictions using a PyTorch model",
          response_description="Prediction results")
async def predict_torch(req: Request) -> PredictResponseTorch:
    """
    Endpoint to make predictions using a PyTorch model.
    """

    request_data = await req.json()

    
    model_data = request_data['model'] # TODO: UNCOMMENT this when proper db is used and data is sent
    # model_data = db[request_data['model']['id']] # TODO: COMMENT OUT when proper db is used and models are uploaded there

    # user_id = request_data['user_id']
    dataset = request_data['dataset']
    dataset_type = dataset['type']

    input_type = dataset['input']['type']
    input_values = dataset['input']['values']

    results = handle_prediction_torch(model_data, input_values)

    return PredictResponseTorch(results=results, message="Successful Prediction")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
