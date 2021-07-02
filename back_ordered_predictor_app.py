from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import pickle
import json
import pickle
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


file = "missing_value_imputer_.pkl"
with open(file, 'rb') as file:
    iterative_imputer = pickle.load(file)

file = "robust_transform.pkl"
with open(file, 'rb') as file:
    robust_scalling = pickle.load(file)

file = "best_model.pkl"
with open(file, 'rb') as file:
    best_model = pickle.load(file)


def function1(X):

    # replacing -99 by NaN in perf_6_month_avg and perf_12_month_avg column
    X.perf_6_month_avg.replace({-99.0: np.nan}, inplace=True)
    X.perf_12_month_avg.replace({-99.0: np.nan}, inplace=True)

    # Converting  Yes and No to 0 and 1 respectively (one hot encoding for categorical features)
    categorical_features = ['rev_stop', 'stop_auto_buy',
                            'ppap_risk', 'oe_constraint', 'deck_risk', 'potential_issue']
    for col in categorical_features:
        X[col].replace({'Yes': 1, 'No': 0}, inplace=True)
        X[col] = X[col].astype(int)

    # handling outliers for real valued features which are very right skwed (analysis from box plot i.e. taking values below 99%tile)
    X = X[(X.national_inv >= 0.000) & (X.national_inv <= 5487.000) &
          (X.forecast_3_month <= 2280.000) & (X.forecast_6_month <= 4335.659999999916) &
          (X.forecast_9_month <= 6316.000) & (X.sales_1_month <= 693.000) & (X.sales_3_month <= 2229.000) &
          (X.sales_6_month <= 4410.000) & (X.sales_9_month <= 6698.000) & (X.min_bank <= 679.6599999999162)]

    # iteraive Imputation (missing value imputation)
    X_array = X.to_numpy()
#     print(X_array)
    X_array = iterative_imputer.transform(X_array)

#     print(X_array)

    # robust scalling on Data

    X_array_robust_scalled = robust_scalling .transform(X_array)

#     print(X_array_robust_scalled)

    predicted_y = best_model.predict(X_array_robust_scalled)


    return predicted_y

def json_validate(data):
    try:
        # giving json data to a variable
        load_value = json.loads(data)
        return "Valid Data"
    except Exception as e:
        print(e)
        return "Invalid"
    else:
        return "Valid"


def dataframe_validate(data):
    try:
        data_value = pd.DataFrame.from_dict(data)
        return "Valid Data"
    except Exception as e:
        print(e)
        return "Invalid"
    else:
        return "Valid"


def feature_validate(df):
    print("dataframe shape : ", df.shape)
    if(df.shape[0] > 0) and (df.shape[1] == 21):
        return "Valid"
    else:
        return "Invalid"


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/data')
def fetch():
    f = open("input_text.txt", "r")
    data = f.read()
    return data


@app.route('/index')
def index():
    return flask.render_template('index (1).html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    data = data['text_data']
    data = data.replace("\'", "\"")
    validity = json_validate(data)
    if validity == "Invalid":
        return jsonify({'Predicted Output': 'Provided string is Invalid'})

    data = json.loads(data)

    df_validity = dataframe_validate(data)
    if df_validity == "Invalid":
        return jsonify({'Predicted Output': 'Provided dataframe is invalid'})

    data = pd.DataFrame.from_dict(data)

    feature_validity = feature_validate(data)
    if feature_validity == "Invalid":
        return jsonify({'Predicted Output': 'Please check if features are more or less than expected.'})

    predict = function1(data)
    if predict == 0:
        output = " Class label is 0 means product is  not on Backordered"
    else:
        output = " Class label is 0 means product is  not on Backordered"

    return jsonify({'prediction': output})


if __name__ == '__main__':
    app.run(debug=True)
