from pydantic import BaseModel, Field
from typing import Literal



#create the root response object.
class RootResponse(BaseModel):
    """"pydantic model for the root response.
    This response gets displayed on the main page."""
    message: str = Field(..., description="Response on the main page.",
                         examples=["Welcome to the root page!"])
    

# create the model response object.
class ModelResponse(BaseModel):
    """pydantic model for the prediction results displayed to the user. 
    This is the flow of data that end users see."""
    got_diagnosis: Literal['M', 'B'] = Field(..., 
                                        description="Model's decision on diagnosis",
                                        examples=["M", "B"])
    
#create the model request object

'''
['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
'''

class ModelRequest(BaseModel):
    """pydantic model for the payload that the model needs for making prediction.
    This payload basically aligns with the structure of the attributes in the training data."""

    radius_mean: float = Field(..., description="radius mean", 
                               examples=[6.98,100.5],ge=6.98)
    texture_mean: float = Field(..., description="texture mean",
                                examples=[9.71,14.50],ge=9.71)
    perimeter_mean: float = Field(..., description="perimeter mean",
                             examples=[44.0,56.74],ge=43.8)
    area_mean: float = Field(..., description="area mean",
                             examples=[144.0,500.2],ge=144.0)
    smoothness_mean: float = Field(..., description="smoothness mean",
                             examples=[0.05,4.50],ge=0.05)
    compactness_mean: float = Field(..., description="compactness mean",
                             examples=[2.1,0.522],ge=0.02)
    concavity_mean: float = Field(..., description="concavity mean",
                             examples=[2.1,0.522],ge=0.0)
    concave_points_mean: float = Field(..., description="concave points mean",
                             examples=[0.67,4.50],ge=0.0)
    symmetry_mean: float= Field(..., description="symmetry mean",
                             examples=[1.40,4.50],ge=0.0)
    fractal_dimension_mean: float = Field(..., description="fractal dimension mean",
                             examples=[2.90,4.50],ge=0.0)
    radius_se: float = Field(..., description="radius se",
                             examples=[0.0078,4.50],ge=0.0)
    texture_se: float= Field(..., description="texture se",
                             examples=[0.78,4.50],ge=0.0)
    perimeter_se: float= Field(..., description="perimeter se",
                             examples=[0.7,5.76],ge=0.0)
    area_se: float = Field(..., description="area_se",
                             examples=[0.89,7.345],ge=0.0)
    smoothness_se: float = Field(..., description="smoothness se",
                             examples=[5.7,6.5],ge=0.0)
    compactness_se: float = Field(..., description="compactness se",
                             examples=[0.78,7.5],ge=0.0)
    concavity_se: float = Field(..., description="concavity se",
                             examples=[2.09,4.578],ge=0.0)
    concave_points_se: float = Field(..., description="concave points se",
                             examples=[2.89,4.5],ge=0.0)
    symmetry_se: float= Field(..., description="symmetry se",
                             examples=[0.67,4.5],ge=0.0)
    fractal_dimension_se: float = Field(..., description="fractal dimension se",
                             examples=[1.98,4.5],gt=0)
    radius_worst: float= Field(..., description="radius worst",
                             examples=[34.50,9.5],ge=0.0)
    texture_worst: float = Field(..., description="texture worst",
                             examples=[12,4.5],ge=0.0)
    perimeter_worst: float= Field(..., description="perimeter",
                             examples=[0.12,4.5],ge=0.0)
    area_worst: float = Field(..., description="area worst",
                             examples=[0.92,4.5],ge=0.0)
    smoothness_worst: float = Field(..., description="smoothness worst",
                             examples=[2.4,5.76],ge=0.0)
    compactness_worst: float= Field(..., description="compactness worst",
                             examples=[3.2,4.5],ge=0.0)
    concavity_worst: float= Field(..., description="concavity worst",
                             examples=[0.72,4.5],ge=0.0)
    concave_points_worst: float = Field(..., description="concave points worst",
                             examples=[0.34,4.5],ge=0.0)
    
    symmetry_worst: float = Field(..., description="symmetry worst",
                             examples=[0.65,14.5],ge=0.0)
    
    fractal_dimension_worst: float= Field(..., description="fractal dimension worst",
                             examples=[0.01,4.5],ge=0.0)
    
    
    
    
    