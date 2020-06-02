import pickle
import os
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FILEPATH = os.path.join(
        os.path.dirname(__file__),
        'model',
        'example1'
    )


def store_model(db):
    # Its important to use binary mode
    dbfile = open(FILEPATH, 'wb')

    # source, destination
    pickle.dump(db, dbfile, pickle.HIGHEST_PROTOCOL)
    dbfile.close()


def extract_model():
    # for reading also binary mode is important
    dbfile = open(FILEPATH, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    lr = get_pipeline()
    lr.set_params(**db)
    return lr


def get_pipeline():
    t = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'), [
            # 'City', 'State', 'Make',
            'Model'
        ]),
        ('scale', StandardScaler(), [
            'Year',
            # 'Mileage'
        ])
    ], remainder='passthrough')

    reg = Pipeline([('transform', t), ('lr', LinearRegression())])
    return reg


def train_model():
    # read dataset
    dtypes = {
        'Year': 'int',
        'Mileage': 'int',
        'City': 'category',
        'State': 'category',
        'Vin': 'category',
        'Make': 'category',
        'Model': 'category',
        'Price': 'int'
    }
    df = pd.read_csv('data/true_car_listings.csv', dtype=dtypes)
    print(df.columns)
    df.head()

    # split dataset
    X = df[[
        'Year',
        # 'Mileage',
        # 'City', 'State', 'Make',
        'Model']]
    y = df.Price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

    reg = get_pipeline()

    model = reg.fit(X_train, y_train)
    print(reg.score(X_train, y_train))
    print(reg.named_steps['lr'].coef_)
    print(reg.named_steps['lr'].intercept_)

    return X_train, X_test, y_train, y_test, reg.get_params()


model_ready = extract_model()

if __name__ == '__main__':
    # dbfile = open(FILEPATH, 'rb')
    # db = pickle.load(dbfile)
    # dbfile.close()
    # print(db)
    #
    # X_train, X_test, y_train, y_test, params = train_model()
    #
    # print(X_test.head())
    # # # testing results
    # lr = get_pipeline()
    # lr.set_params(**params)
    # prediction = lr.predict(X_test)
    # print(lr.score(X_test, y_test))
    # print(r2_score(prediction, y_test))
    # print(mean_squared_error(prediction, y_test))
    #
    # store_model(params)

    df = DataFrame(data=[[2011, 'Escape4WD']], columns=['Year', 'Model'])
    print(df.head())
    model_ready.predict(df)



    #
    # # store results
    # dict_model = {
    #     "coef_": reg.coef_,
    #     "intercept_": reg.intercept_
    # }
    # store_model(dict_model)
