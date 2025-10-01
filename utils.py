from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,LabelEncoder
from schema import ModelRequest
import pandas as pd
import numpy as np
import logging
from logging import getLogger
from zenml.client import Client
from typing import List,Tuple, Dict
import os
from dotenv import load_dotenv 

load_dotenv()

MODEL_URI = os.getenv("MODEL_URI")
ENCODER_URI = os.getenv("ENCODER_URI")
SCALER_URI = os.getenv("SCALER_URI")


#configure the logging
logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

#function for getting artifact.

def get_artifacts() -> Tuple[LabelEncoder, StandardScaler,
                             LogisticRegression]:
    artifact_uri = [ENCODER_URI, SCALER_URI, MODEL_URI]
    artifact_list =[]
    for uri in artifact_uri:
        artifact = Client().get_artifact_version(str(uri))
        artifact_object = artifact.load()
        artifact_list.append(artifact_object)

    return tuple(artifact_list)

#write a function that makes prediction
def predict_diagnosis(data:Dict) -> str:
    """It takes payload as dict, processes it with scaler+model
    and returns diagnosis label('M' or 'B').
    """
    encoders, scaler, model = get_artifacts()

    target_encoder = encoders['diagnosis']

    #Ensure column order matches training schema
    features = list(ModelRequest.model_fields.keys())
    input_df = pd.DataFrame([data],columns=features)


    # Rename columns to match model's training data
    input_df.rename(columns= {
        "concave_points_mean": "concave points_mean",
        "concave_points_se": "concave points_se",
        "concave_points_worst": "concave points_worst"
    }, inplace=True)
    
    for col in encoders.keys():
        if col != 'diagnosis':
            input_df[col] = encoders[col].transform(input_df[col])

            #Reorder columns to match training
        training_features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
        'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 
        'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
        'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
        'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    input_df = input_df[training_features]


    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=training_features)


    #Model prediction (numeric 0/1)
    pred = model.predict(input_scaled_df) [0]

    #Map back to original label
    target_enconder = encoders['diagnosis']
    diagnosis = target_encoder.inverse_transform([pred]) [0]

    

    logger.info(f'Prediction:{diagnosis}, input:{data}')

    return diagnosis
