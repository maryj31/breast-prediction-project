🩺 Breast Cancer Prediction

Introduction

This project applies machine learning to predict whether a breast tumor is Malignant (M) or Benign (B) using the Breast Cancer Wisconsin (Diagnostic) dataset from Kaggle.
The goal is to assist in early cancer detection, which plays a crucial role in successful treatment.


 Dataset

Source: Kaggle – Breast Cancer Wisconsin (Diagnostic) Dataset

Features: 30 numerical attributes such as radius, texture, perimeter, smoothness, concavity, and symmetry.

Target Variable:

M → Malignant (cancerous)

B → Benign (non-cancerous)


 Workflow

1. Data Preprocessing

Cleaned dataset

Normalized feature values

Encoded target (M/B) internally as numeric labels for model training



2. Model Training

Model training with Logistic Regression 

Ensured high f1 score 

3. Pipeline (ZenML)

Created ML pipeline for preprocessing, training, and evaluation

Metrics logged with Logfire for monitoring

Model versioned and tracked



4. Deployment (FastAPI)

Exposed a FAST API endpoint /get_diagnosis

Accepts patient features and returns prediction (M or B)



🚀 How to Run

⿡ Clone Repository

git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction

⿢ Create Virtual Environment

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

⿣ Install dependencies


⿤ Run Pipeline

uv run pipeline.py

⿥ Start FastAPI App

uvicorn main:app --reload

Swagger UI → http://127.0.0.1:8000/docs

Redoc → http://127.0.0.1:8000/redoc



📈 Results

F1-score : 0.94



Reliable performance in distinguishing between malignant and benign tumors.


 Example API Request

 Request (to /get_diagnosis)

{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.0787
}

 Response


  "got_diagnosis": "M",
  



 Future Work

Add deep learning models (e.g., Neural Networks)

Add interpretability with SHAP/LIME

Containerize with Docker for production deployment

Integrate continuous monitoring of predictions


 Acknowledgements

Kaggle – Breast Cancer Wisconsin Dataset

scikit-learn

ZenML

Author 

This project was developed by [Suleiman Mariam] to showcase the end to end deployment 
of a machine learning model using FastAPI and ZenML, serving both as a healthcare focused 
solution and a demonstration of practical MLOps skills.
