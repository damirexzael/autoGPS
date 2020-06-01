import pickle
from random import random
import os

FILEPATH = os.path.join(
        os.path.dirname(__file__),
        'model',
        'example1'
    )


class Example:
    def __init__(self, **kwargs):
        print("init")
        print(kwargs)

    def predict(self, **kwargs):
        print(kwargs)
        return random()


def store_model(db):
    # Its important to use binary mode
    dbfile = open(FILEPATH, 'wb')

    # source, destination
    pickle.dump(db, dbfile, pickle.HIGHEST_PROTOCOL)
    dbfile.close()


def load_model():
    # for reading also binary mode is important
    dbfile = open(FILEPATH, 'rb')
    db = pickle.load(dbfile)
    dbfile.close()
    return db


def extract_model():
    params = load_model()
    model = Example(**params)
    return model


model_ready = extract_model()

if __name__ == '__main__':
    parameters = {
        "dad": 123,
        "qwe": 338
    }
    store_model(parameters)
    model_ready = extract_model()
    print(parameters)
    print(model_ready.predict())
