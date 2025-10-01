from zenml import pipeline
from steps.data_cleaner import encode_dataset, scale_dataset, split_dataset,drop_columns
from steps.dataloader import load_data
from steps.training import train_model
from zenml.logger import get_logger
from typing_extensions import Annotated
from typing import Optional
from sklearn.linear_model import LogisticRegression



#configure logging
logger = get_logger(__name__)


@pipeline(enable_cache=False) 
def breast_pipeline():
    data = load_data()
    data = drop_columns(data)
    data, label_encoders = encode_dataset(data)
    X_train, X_test, y_train, y_test = split_dataset(data)
    X_train, X_test, scaler = scale_dataset(X_train, X_test)
    model = train_model(X_train, X_test, y_train, y_test)

    

if __name__ == "__main__":
    breast_pipeline()

