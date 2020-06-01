from analytics.libs.model_administrator import model_ready


def predict_model(**kwargs):
    print("predict_model", kwargs)
    return model_ready.predict(**kwargs)


if __name__ == '__main__':
    print(predict_model())
