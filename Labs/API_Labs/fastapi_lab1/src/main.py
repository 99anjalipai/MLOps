from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    label: str

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]

        prediction = predict_data(features)
        pred_idx = int(prediction[0])
        return IrisResponse(label=TARGET_NAMES[pred_idx])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
