import os
import pickle

import  pandas as pd
from sklearn.pipeline import Pipeline


class MachineLearning:
    is_load = False
    pipeline = Pipeline

    def __init__(self, filename):
        print("init")
        self.params = Params(filename)

    def load_model(self):
        self.pipeline = self.params.get_pipeline()
        self.is_load = True

    def get_features(self):
        features = list()
        for trasform in self.pipeline.steps[0][1].transformers:
            features = features + trasform[2]
        print(features)
        return features

    def predict_df(self, df):
        if not self.is_load:
            raise ValueError("First load model")
        # df = df[self.get_features()]
        print(df)
        return self.pipeline.predict(df[['Mileage', 'Model']])

    def predict_dict(self, dict_values):
        print(dict_values)
        df = pd.DataFrame([dict_values])
        return self.predict_df(df)[0]


class Params:
    def __init__(self, pipeline_name):
        self.path = os.path.join(os.path.dirname(__file__), 'model')
        self.pipeline = pipeline_name

    def get_pipeline(self):
        return self.__load_pickle(self.pipeline)

    def set_pipeline(self, params):
        self.__save_pickle(params, self.pipeline)

    def __load_pickle(self, filename):
        dbfile = open(f'{self.path}/{filename}', 'rb')
        data = pickle.load(dbfile)
        dbfile.close()
        return data

    def __save_pickle(self, data, filename):
        # Its important to use binary mode
        dbfile = open(f'{self.path}/{filename}', 'wb')
        # source, destination
        pickle.dump(data, dbfile, pickle.HIGHEST_PROTOCOL)
        dbfile.close()


if __name__ == '__main__':
    ml = MachineLearning('prod_pipeline')
    ml.load_model()

    # df = pd.DataFrame(data=[[19650, 'TacomaAccess']], columns=['Mileage', 'Model'])
    # print(df.head())
    # a = ml.predict_df(df)
    # print(a)

    print(ml.get_features())
    dictionary = {
        "Mileage": "19650",
        "Model": "TacomaAccess"
    }
    print(ml.predict_dict(dictionary))

    # param_grid = {
    #     'classifier__n_estimators': [200, 500],
    #     'classifier__max_features': ['auto', 'sqrt', 'log2'],
    #     'classifier__max_depth': [4, 5, 6, 7, 8],
    #     'classifier__criterion': ['gini', 'entropy']}
    # from sklearn.model_selection import GridSearchCV
    #
    # CV = GridSearchCV(rf, param_grid, n_jobs=1)
    #
    # CV.fit(X_train, y_train)
    # print(CV.best_params_)
    # print(CV.best_score_)

    # t = ColumnTransformer(transformers=[
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'), [
    #         # 'City', 'State', 'Make',
    #         'Model'
    #     ]),
    #     ('scale', StandardScaler(), [
    #         # 'Year',
    #         'Mileage'
    #     ])
    # ], remainder='passthrough')
