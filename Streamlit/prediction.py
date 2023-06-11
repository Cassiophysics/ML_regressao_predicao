import pickle
def predict(data):
    xgbr2 = pickle.load(open('saved_steps.pkl', 'rb'))
    return xgbr2.predict(data)