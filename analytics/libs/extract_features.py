from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analytics.libs.machine_learning import MachineLearning


def extract_features(dict_values):
    """
    Check if all the
    :param kwargs:
    :return:
    """
    print("extract_features", dict_values)


    return 5.3


if __name__ == '__main__':
    ml = MachineLearning()
    ml.load_model()
    print(ml.pipeline.n_features_in_)
    t = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), [
            # 'City', 'State', 'Make',
            'Model'
        ]),
        ('scale', StandardScaler(), [
            # 'Year',
            'Mileage'
        ])
    ], remainder='drop')
    # t.get_feature_names()
    print(ml.pipeline.named_steps['transform'].transformers)
