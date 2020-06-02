from pandas import DataFrame

from analytics.libs.model_administrator import model_ready


def predict_model(**kwargs):
    print("predict_model", kwargs)
    year = kwargs.get('year')
    model = kwargs.get('model')
    df = DataFrame(data=[[year, model]], columns=['Year', 'Model'])
    print(df.head())

    return model_ready.predict(df)


if __name__ == '__main__':
    values = {
        "year": 2011,
        "model": 'Escape4WD'
    }
    print(predict_model(**values))
