import joblib
import numpy as np
import pandas as pd
import xgboost


class Predictor(object):
    def __init__(self):
        self.colname = [
            "gender",
            "age",
            "hypertension",
            "heart_disease",
            "ever_married",
            "work_type",
            "Residence_type",
            "avg_glucose_level",
            "bmi",
            "smoking_status",
        ]
        model_path = "model/model.sav"
        self._model = joblib.load(model_path)
        self.threshold = 0.3

    def predict_proba(self, instances):
        inputs = np.asarray(instances).reshape(1, -1)
        inputs = xgboost.DMatrix(pd.DataFrame(inputs, columns=self.colname))
        prediction = self._model.predict(inputs)[0]
        return prediction

    def predict(self, instances):
        proba = self.predict_proba(instances)
        prediction = np.where(proba > 0.3, 1, 0)
        return prediction
