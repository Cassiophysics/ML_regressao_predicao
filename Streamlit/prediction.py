#import joblib
#def predict(data):
#    xgbr2 = joblib.load('xgbr2_model.sav')
#    return xgbr2.predict(data)

import pickle
def predict(data):
    xgbr2 = pickle.load(open('modelo_xgbr2.pkl', 'rb'))
    return xgbr2.predict(data)