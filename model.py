from flask import Flask, request
import pickle
import pandas as pd

app = Flask(__name__)


@app.route('/prediction', methods=['POST'])
def predict_model():
    if request.method == 'POST':
        data = request.json  # a multidict containing POST data

        dict_value = {'CODE_GENDER_M': data['CODE_GENDER_M'],
                      'EMERGENCYSTATE_MODE_Yes': data['EMERGENCYSTATE_MODE_Yes'],
                      'OCCUPATION_TYPE_Cleaning staff': data['OCCUPATION_TYPE_Cleaning staff'],
                      'OCCUPATION_TYPE_Cooking staff': data['OCCUPATION_TYPE_Cooking staff'],
                      'OCCUPATION_TYPE_Core staff': data['OCCUPATION_TYPE_Core staff'],
                      'OCCUPATION_TYPE_Drivers': data['OCCUPATION_TYPE_Drivers'],
                      'OCCUPATION_TYPE_HR staff': data['OCCUPATION_TYPE_HR staff'],
                      'OCCUPATION_TYPE_High skill tech staff': data['OCCUPATION_TYPE_High skill tech staff'],
                      'OCCUPATION_TYPE_IT staff': data['OCCUPATION_TYPE_IT staff'],
                      'OCCUPATION_TYPE_Laborers': data['OCCUPATION_TYPE_Laborers'],
                      'OCCUPATION_TYPE_Low-skill Laborers': data['OCCUPATION_TYPE_Low-skill Laborers'],
                      'OCCUPATION_TYPE_Managers': data['OCCUPATION_TYPE_Managers'],
                      'OCCUPATION_TYPE_Medicine staff': data['OCCUPATION_TYPE_Medicine staff'],
                      'OCCUPATION_TYPE_Private service staff': data['OCCUPATION_TYPE_Private service staff'],
                      'OCCUPATION_TYPE_Realty agents': data['OCCUPATION_TYPE_Realty agents'],
                      'OCCUPATION_TYPE_Sales staff': data['OCCUPATION_TYPE_Sales staff'],
                      'OCCUPATION_TYPE_Secretaries': data['OCCUPATION_TYPE_Secretaries'],
                      'OCCUPATION_TYPE_Security staff': data['OCCUPATION_TYPE_Security staff'],
                      'OCCUPATION_TYPE_Waiters/barmen staff': data['OCCUPATION_TYPE_Waiters/barmen staff'],
                      'WALLSMATERIAL_MODE_Mixed': data['WALLSMATERIAL_MODE_Mixed'],
                      'WALLSMATERIAL_MODE_Monolithic': data['WALLSMATERIAL_MODE_Monolithic'],
                      'WALLSMATERIAL_MODE_Others': data['WALLSMATERIAL_MODE_Others'],
                      'WALLSMATERIAL_MODE_Panel': data['WALLSMATERIAL_MODE_Panel'],
                      'WALLSMATERIAL_MODE_Stone,brick': data['WALLSMATERIAL_MODE_Stone,brick'],
                      'WALLSMATERIAL_MODE_Wooden': data['WALLSMATERIAL_MODE_Wooden'],
                      'EXT_SOURCE_3': data['EXT_SOURCE_3'],
                      'REGION_RATING_CLIENT': data['REGION_RATING_CLIENT'],
                      'AMT_GOODS_PRICE': data['AMT_GOODS_PRICE'],
                      'GOODS_PRICE_CREDIT_PER': data['GOODS_PRICE_CREDIT_PER'],
                      'DAYS_WORKING_PER': data['DAYS_WORKING_PER'],
                      'ANNUITY_DAYS_BIRTH_PERC': data['ANNUITY_DAYS_BIRTH_PERC']}

        X_new = pd.DataFrame(dict_value, index=[0])

        print('data received', data)
        print('dataframe created', X_new)
        file_name = "predict_loan_GBC.pkl"

        # load the pickle
        model = pickle.load(open(file_name, "rb"))
        # work in progress : entrainer le model avec les parramètres et les rajouter
        # prediction = random.random()
        prediction = model.predict(X_new)

        return f"The prediction for this individual is {round(prediction[0], 2)}!"


if __name__ == "__main__":
    app.run()
