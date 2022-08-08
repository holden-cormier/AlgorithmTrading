import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib as plt
from DNNModel import *

if __name__ == "__main__":
    file_data = pd.read_csv("UpdateData.csv", parse_dates=["time"], index_col="time")
    instrument = file_data.columns[0]
    file_data["returns"] = np.log(file_data[instrument] / file_data[instrument].shift())

    # creating features to determine position
    df = file_data.copy()
    df["dir"] = np.where(df["returns"] > 0, 1, 0)
    # simple moving average feature
    min_window = 50
    max_window = 150
    df["sma"] = df[instrument].rolling(min_window).mean() - df[instrument].rolling(max_window).mean()
    # bollinger bands feature
    df["boll"] = (df[instrument] - df[instrument].rolling(min_window).mean()) / (df[instrument].rolling(min_window).std())
    # rolling min vs current feature
    df["min"] = df[instrument].rolling(min_window).min() / df[instrument] - 1
    # rolling max vs current feature
    df["max"] = df[instrument].rolling(min_window).max() / df[instrument] - 1
    # momentum based feature
    df["mom"] = df["returns"].rolling(3).mean()
    # volitility based feature
    df["vol"] = df["returns"].rolling(min_window).std()

    df.dropna(inplace=True)

    # adding lags
    lags = 5
    cols = []
    features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]
    for feature in features:
        for lag in range(1, lags + 1):
            col = "{}_lag_{}".format(feature, lag)
            df[col] = df[feature].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)

    # creating training and test data sets
    div = int(len(df) * 0.8)
    train = df.iloc[:div].copy()
    test = df.iloc[div:].copy()
    mu = train.mean()
    std = train.std()
    # standardize training set
    train_s = (train - mu) / std
    set_seeds(100)
    model = create_model(hl=3, hu=100, dropout=True, input_dim=len(cols))
    model.fit(x=train_s[cols], y=train["dir"], epochs=50, verbose=False, validation_split=0.2, shuffle=False, class_weight=cw(train))
    print(model.evaluate(train_s[cols], train["dir"]))

    model.save("DNN_model")
    params = {"mu": mu, "std": std}
    pickle.dump(params, open("params.pkl", "wb"))
    print("Model finished training")






























