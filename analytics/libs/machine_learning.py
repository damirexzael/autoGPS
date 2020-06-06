import os
import pandas as pd
from sklearn.pipeline import Pipeline
from joblib import dump, load


class MachineLearning:
    is_load = False
    pipeline = Pipeline
    features = list()

    def __init__(self, filename):
        print("init")
        self.params = Params(filename)

    def load_model(self):
        model = self.params.get_model()
        self.features = ['Mileage', 'Model']
        self.pipeline = model
        self.is_load = True

    def predict_df(self, df):
        if not self.is_load:
            raise ValueError("First load model")
        # df = df[self.get_features()]
        print(df)
        print(df[self.features])
        return self.pipeline.predict(df[self.features])

    def predict_dict(self, dict_values):
        print(dict_values)
        df = pd.DataFrame([dict_values])
        return self.predict_df(df)[0]


class Params:
    def __init__(self, pipeline_name):
        self.path = os.path.join(os.path.dirname(__file__), 'model')
        self.pipeline = pipeline_name

    def get_model(self):
        return load(f'{self.path}/{self.pipeline}.joblib')

    def set_model(self, model):
        dump(model, f'{self.path}/{self.pipeline}.joblib')


if __name__ == '__main__':
    ml = MachineLearning('test_pipeline')
    ml.load_model()

    print(ml.features)
    dictionary = {
        "Mileage": "19650",
        "Model": "TacomaAccess"
    }
    print(ml.predict_dict(dictionary))
