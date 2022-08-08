import pandas as pd
import numpy as np
import tpqoa
import fxcmpy
from datetime import datetime, timedelta
import time
import pickle
from DNNModel import *
import keras


class Trader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, window, lags, model, mu, std, units):
        super().__init__(conf_file)
        self.instrument = instrument
        self.bar_length = pd.to_timedelta(bar_length)
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None
        self.last_bar = None
        self.units = units
        self.position = 0
        self.profits = []

        self.window = window
        self.lags = lags
        self.model = model
        self.mu = mu
        self.std = std
        self.original_units = units

    def get_most_recent(self, days=5):
        while True:
            time.sleep(2)
            now = datetime.utcnow()
            now = now - timedelta(microseconds=now.microsecond)
            past = now - timedelta(days=days)
            df = self.get_history(instrument=self.instrument, start=past, end=now,
                                  granularity="S5", price="M", localize=False).c.dropna().to_frame()
            df.rename(columns={"c": self.instrument}, inplace=True)
            df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
            self.raw_data = df.copy()
            self.last_bar = self.raw_data.index[-1]
            if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
                self.start_time = pd.to_datetime(datetime.utcnow()).tz_localize("UTC")
                break

    def on_success(self, time, bid, ask):
        if self.ticks == 1:
            print("Trader is running")
        if self.ticks % 100 == 0 or self.ticks <= 10:
            print(self.ticks, end=" ")

        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument: (ask + bid) / 2},
                          index=[recent_tick])
        self.tick_data = pd.concat([self.tick_data, df])
        #self.tick_data = self.tick_data.append(df)

        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()

    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length, label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]

    def define_strategy(self):
        df = self.raw_data.copy()

        df = df.append(self.tick_data)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        df["dir"] = np.where(df["returns"] > 0, 1, 0)
        df["sma"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
        df["boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[
            self.instrument].rolling(self.window).std()
        df["min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
        df["max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
        df["mom"] = df["returns"].rolling(3).mean()
        df["vol"] = df["returns"].rolling(self.window).std()
        df.dropna(inplace=True)

        # create lags
        self.cols = []
        features = ["dir", "sma", "boll", "min", "max", "mom", "vol"]

        for f in features:
            for lag in range(1, self.lags + 1):
                col = "{}_lag_{}".format(f, lag)
                df[col] = df[f].shift(lag)
                self.cols.append(col)
        df.dropna(inplace=True)

        # standardization
        df_s = (df - self.mu) / self.std
        # predict
        print(df[self.cols])
        df["proba"] = self.model.predict(df_s[self.cols])

        # determine positions
        df = df.loc[self.start_time:].copy()
        df["position"] = np.where(df.proba < 0.485, -1, np.nan)
        df["position"] = np.where(df.proba > 0.515, 1, df.position)
        df["position"] = df.position.ffill().fillna(0)

        self.units = self.size_position(probability=df["proba"][-1], position_size=self.original_units)
        self.data = df.copy()

    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "Opening long position")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "Opening long position")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "Opening short position")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "Opening short position")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "selling current position")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "selling current position")
            self.position = 0

    def report_trade(self, order, going):
        time = order["time"]
        units = order["units"]
        price = order["price"]
        pl = float(order["pl"])
        self.profits.append(pl)
        cumpl = sum(self.profits)
        print("\n" + 100 * "-")
        print("{} | {}".format(time, going))
        print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
        print(100 * "-" + "\n")

    def size_position(self, probability, position_size):
        if probability > .525:
            return position_size
        elif probability > 515:
            return position_size * 0.5
        elif probability < .485:
            return position_size * 0.5
        elif probability < .475:
            return position_size
        else:
            return position_size


if __name__ == '__main__':
    model = keras.models.load_model("DNN_model")
    params = pickle.load(open("params.pkl", "rb"))
    mu = params["mu"]
    std = params["std"]
    trader = Trader("oanda.cfg", "EUR_USD", bar_length="30min", window=50, lags=5, model=model, mu=mu, std=std, units=100000)
    trader.get_most_recent()
    trader.stream_data(trader.instrument, stop=100000)
    if trader.position != 0:
        close_order = trader.create_order(trader.instrument, units=-trader.position * trader.units,
                                          suppress=True, ret=True)
        trader.report_trade(close_order, "Closing all positions")
        trader.position = 0

