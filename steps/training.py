from zenml import step
from typing import Optional
from typing_extensions import Annotated, Tuple, Dict
import pandas as pd
from zenml.logger import get_logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score



#configure logging
logger = get_logger(__name__)


#training step
@step
def train_model(X_train:pd.DataFrame, X_test:pd.DataFrame,
                 y_train:pd.Series, y_test:pd.Series) -> Annotated[Optional[LogisticRegression],
                                               "Base Model"]:
    """This step trains the base model and returns the model object."""


    model = None
    try:
        model = LogisticRegression(random_state=23, class_weight='balanced')
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
    

        #compute the metrics
        train_f1 = f1_score(y_train, train_preds)
        test_f1 = f1_score(y_test, test_preds)
        logger.info(f"""
                    Training completed with the following metrics: \n
                    train f1 score: {train_f1}
                    test f1 score: {test_f1}
                    """)
    except Exception as e:
        logger.error(f"An error occured. Detail: {e}")

    return model
                                               
                                               