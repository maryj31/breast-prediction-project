from zenml import step
from typing import Optional
from typing_extensions import Annotated
import kagglehub
import pandas as pd
import os
from zenml.logger import get_logger

#configure logging
logger = get_logger(__name__)


#define the data loading step
@step
def load_data() -> Annotated[Optional[pd.DataFrame], "Full Dataset"]:
    """This step takes the training data from kagglehub and load it into the 
    training pipeline."""
    data = None
    try:
        # download dataset
        path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")
        file_name = os.listdir(path)[0]
        data_url = os.path.join(path, file_name)
        logger.info(f"found data file in the filepath:{path} with name{file_name}")
        data = pd.read_csv(data_url)
        logger.info(f"""
                    Data loaded successfully with the following attribute:
                    {data.info()}
                    """)
    except Exception as e:
        logger.error(f"An error occured. Detail: {e}")
        raise e
    
    return data