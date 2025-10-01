from zenml import step
from typing import Optional
from typing_extensions import Annotated, Tuple, Dict
import pandas as pd
import os
from zenml.logger import get_logger
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#configure logging
logger = get_logger(__name__)

#drop ID, encode cat, scale the dataset, split .
@step
def drop_columns(data: pd.DataFrame) -> Annotated[Optional[pd.DataFrame],
                                                  "Dropped Redundant Columns"]:
    """This steps dropped unwanted column that is not important to the scope of this project."""
    try:
        data.drop(columns=["id"],inplace= True)
        logger.info(f"Column dropped successfully. We now have the following:\n{data.columns}")
    except Exception as e:
        logger.error(f"An error occured. Detail: {e}")
        raise e
    
    return data

# encode y
@step
def encode_dataset(data: pd.DataFrame) -> Tuple[
    Annotated[Optional[pd.DataFrame], "Encoded Data"],
    Annotated[Optional[Dict[str,LabelEncoder]], "Label Encoders"]]:

    """This step encodes the target column(diagnosis) in the dataset and returns both the dataset
    and a dictionary of encoder object"""

    try:
        encoder = LabelEncoder()
        data["diagnosis"] = encoder.fit_transform(data["diagnosis"])
        logger.info(f""" 
                    Encoded diagnosis successfully, new data info: \n
                    {data.info()}
                    """)
    except Exception as e:
        logger.error(f"An error occured. Detail: {e}")
        raise e

    return data, {"diagnosis":encoder}
                                          

 # split the dataset

@step
def split_dataset(data: pd.DataFrame) -> Tuple[
    Annotated[Optional[pd.DataFrame], "X_train"],
    Annotated[Optional[pd.DataFrame], "X_test"],
    Annotated[Optional[pd.Series], "y_train"], 
    Annotated[Optional[pd.Series], "y_test"]]:
    """This step splits the dataset into train set and test set"""    
    X_train, X_test, y_train, y_test = None, None, None, None
    try:
        X = data.drop(columns = ["diagnosis"])
        y = data["diagnosis"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state = 23,stratify=y)

        logger.info(f"""
                    Splitted the dataset successfully.\n
                    X_train: {X_train.shape}
                    X_test: {X_test.shape}
                    y_train:{y_train.shape}
                    y_test:{y_test.shape}
                    """)

    except Exception as e:
        logger.error(f"An error occured. Detail {e}")

    return X_train, X_test, y_train, y_test
    
                                                            
# scale the dataset
@step
def scale_dataset(X_train: pd.DataFrame,
                  X_test: pd.DataFrame) -> Tuple[
                      Annotated[Optional[pd.DataFrame], "Scaled X_train"],
                      Annotated[Optional[pd.DataFrame], "Scaled X_test"],
                      Annotated[Optional[StandardScaler], "Scaler Object"]]:
    scaler = None

        
    try:
        scaler = StandardScaler()
        scaler.fit(X_train)
        columns = list(X_train.columns)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # convert back to dataframe
        X_train = pd.DataFrame(data=X_train_scaled, columns=columns)
        X_test = pd.DataFrame(data=X_test_scaled, columns=columns)
        logger.info("Scaling completed!")

    except Exception as e:
        logger.error(f"An error occured. Detail: {e}")

    return X_train, X_test, scaler



                  
                           