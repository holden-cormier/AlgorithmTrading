import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle


if __name__ == "__main__":
    data = pd.read_csv("five_minute.csv", parse_dates=["time"], index_col="time")
    data["returns"] = np.log(data.div(data.shift(1)))
    data.dropna(inplace=True)
    data["direction"] = np.sign(data.returns)
    lags = 4
    cols = []
    for lag in range(1, lags + 1):
        col = "lag{}".format(lag)
        data[col] = data.returns.shift(lag)
        cols.append(col)
    data.dropna(inplace=True)
    lm = LogisticRegression(C = 1e6, max_iter = 100000, multi_class = "ovr")
    lm.fit(data[cols], data.direction)
    pickle.dump(lm, open("logreg.pkl", "wb"))
    exit(0)





