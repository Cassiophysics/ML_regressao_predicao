import joblib
def predict(data):
    xgbr2 = joblib.load('xgbr2_model.sav')
    return xgbr2.predict(data)

#import joblib
#def predict(data):
#    xgbr2 = joblib.load(open('saved_steps.pkl', 'rb'))
#    return xgbr2.predict(data)