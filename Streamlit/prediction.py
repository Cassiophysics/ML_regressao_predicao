import pickle
def predict(data):
    xgbr2 = pickle.load(open('modelo_xgbr2.pkl', 'rb'))
    return xgbr2.predict(data)