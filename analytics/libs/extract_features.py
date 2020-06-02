def extract_features(**kwargs):
    """
    Check if all the
    :param kwargs:
    :return:
    """
    print("extract_features", kwargs)
    values = {
        'year': int(kwargs.get('year', 2011)),
        'model': kwargs.get('model', 'Escape4WD')
    }
    print("values", values)
    return values
