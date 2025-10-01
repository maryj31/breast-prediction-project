from fastapi import FastAPI, HTTPException, status
from sklearn.linear_model import LogisticRegression
from schema import RootResponse, ModelRequest, ModelResponse
import uvicorn
from utils import  predict_diagnosis
import logfire
from dotenv import load_dotenv
import os
import uvicorn


#configure the environment variables and monitoring
load_dotenv()

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

#init the app
app = FastAPI(
    title="Breast Prediction ML App",
    version= "v1-base"
)

#logfire config
logfire.configure(token=LOGFIRE_TOKEN)
logfire.instrument_fastapi(app=app)



#create the root endpoint

@app.get(path="/", tags=["Home"], response_model= RootResponse)
async def root():
    """This endpoint is the root endpoint that gets called 
    the first time the app is launched."""
    return RootResponse(message = "We are live")



#create the prediction point
@app.post(path="/get_diagnosis", tags=["Diagnosis"], response_model=ModelResponse)
def get_diagnosis(payload: ModelRequest):
    """This is the main endpoint of diagnosing breast.
    This endpoint takes a payload of breast measurements and return the prediction of growth."""
    

    try:
        data = payload.model_dump()
        diagnosis = predict_diagnosis(data)
        logfire.info(f"got_diagnosis:{diagnosis}, payload: {data}")



        return ModelResponse(
            got_diagnosis = diagnosis
        )

    except Exception as err:
        logfire.error(f"An error occured. Details: {err}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"{err}")



if __name__ == "__main__":
    uvicorn.run("main:app", port=8080, host="localhost", reload=True)
