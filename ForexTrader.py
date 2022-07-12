import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import tpqoa
from datetime import datetime, timedelta
import time
import pickle


# class ForexTrader(tpqoa.tpqoa):
#     def __init__(self, conf_file, instrument, bar_length, units):
#         super().__init__(conf_file)
#         self.instrument = instrument
#         self.bar_length = pd.to_timedelta(bar_length)
#         self.tick_data = pd.DataFrame()
#         self.raw_data = None
#         self.data = None
#         self.last_bar = None
#         self.units = units
#         self.positions = 0
#         self.profits = None
#
#         machine_learning = MachineLearning()
#         self.lags = machine_learning.lags
#         self.model = pickle.load(open("logreg.pkl", "rb"))
#
#     def get_most_recent(self, days=5):
#         while True:
#             time.sleep(2)
#             now = datetime.utcnow()
#             now = now - timedelta(microseconds=now.microsecond)
#             past = now - timedelta(days=days)
#             df = self.get_history(instrument=self.instrument, start=past, end=now,
#                                   granularity="S5", price="M", localize=False).c.dropna().to_frame()
#             df.rename(columns={"c": self.instrument}, inplace=True)
#             df = df.resample(self.bar_length, label="right").last().dropna().iloc[:-1]
#             self.raw_data = df.copy()
#             self.last_bar = self.raw_data.index[-1]
#             if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
#                 break
#
#     def on_success(self, time, bid, ask):
#         print(self.ticks, end=" ")
#
#         recent_tick = pd.to_datetime(time)
#         df = pd.DataFrame({self.instrument: (ask + bid) / 2},
#                           index=[recent_tick])
#         self.tick_data = self.tick_data.append(df)
#
#         if recent_tick - self.last_bar > self.bar_length:
#             self.resample_and_join()
#             self.define_strategy()
#             self.execute_trades()
#
#     def resample_and_join(self):
#         self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length,
#                                                                      label="right").last().ffill().iloc[:-1])
#         self.tick_data = self.tick_data.iloc[-1:]
#         self.last_bar = self.raw_data.index[-1]
#
#     def define_strategy(self):
#         df = self.raw_data.copy()
#
#         df = df.append(self.tick_data)
#         df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
#         cols = []
#         for lag in range(1, self.lags + 1):
#             col = "lag{}".format(lag)
#             df[col] = df.returns.shift(lag)
#             cols.append(col)
#         df.dropna(inplace=True)
#         df["position"] = self.model.predict(df[cols])
#
#         self.data = df.copy()
#
#     def execute_trades(self):
#         if self.data["position"].iloc[-1] == 1:
#             if self.position == 0:
#                 order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
#                 self.trade_info(order, "GOING LONG")
#             elif self.position == -1:
#                 order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
#                 self.trade_info(order, "GOING LONG")
#             self.position = 1
#         elif self.data["position"].iloc[-1] == -1:
#             if self.position == 0:
#                 order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
#                 self.trade_info(order, "GOING SHORT")
#             elif self.position == 1:
#                 order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
#                 self.trade_info(order, "GOING SHORT")
#             self.position = -1
#         elif self.data["position"].iloc[-1] == 0:
#             if self.position == -1:
#                 order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
#                 self.trade_info(order, "GOING NEUTRAL")
#             elif self.position == 1:
#                 order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
#                 self.trade_info(order, "GOING NEUTRAL")
#             self.position = 0
#
#     def trade_info(self, order, going):
#         time = order["time"]
#         units = order["units"]
#         price = order["price"]
#         pl = float(order["pl"])
#         self.profits.append(pl)
#         cumpl = sum(self.profits)
#         print("\n" + 100 * "-")
#         print("{} | {}".format(time, going))
#         print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
#         print(100 * "-" + "\n")
#
#
# def main():
#     print("works")
#     trader = ForexTrader("oanda.cfg", "EUR_USD", "5min", units=100000)
#     print("works")
#

class MLTrader(tpqoa.tpqoa):
    def __init__(self, conf_file, instrument, bar_length, lags, model, units):
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

        # *****************add strategy-specific attributes here******************
        self.lags = lags
        self.model = model
        # ************************************************************************

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
                break

    def on_success(self, time, bid, ask):
        print(self.ticks, end=" ")

        recent_tick = pd.to_datetime(time)
        df = pd.DataFrame({self.instrument: (ask + bid) / 2},
                          index=[recent_tick])
        self.tick_data = self.tick_data.append(df)

        if recent_tick - self.last_bar > self.bar_length:
            self.resample_and_join()
            self.define_strategy()
            self.execute_trades()

    def resample_and_join(self):
        self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length,
                                                                     label="right").last().ffill().iloc[:-1])
        self.tick_data = self.tick_data.iloc[-1:]
        self.last_bar = self.raw_data.index[-1]

    def define_strategy(self):  # "strategy-specific"
        df = self.raw_data.copy()

        # ******************** define your strategy here ************************
        df = df.append(self.tick_data)  # append latest tick (== open price of current bar)
        df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
        cols = []
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            df[col] = df.returns.shift(lag)
            cols.append(col)
        df.dropna(inplace=True)
        df["position"] = lm.predict(df[cols])
        # ***********************************************************************

        self.data = df.copy()

    def execute_trades(self):
        if self.data["position"].iloc[-1] == 1:
            if self.position == 0:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
            elif self.position == -1:
                order = self.create_order(self.instrument, self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING LONG")
            self.position = 1
        elif self.data["position"].iloc[-1] == -1:
            if self.position == 0:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units * 2, suppress=True, ret=True)
                self.report_trade(order, "GOING SHORT")
            self.position = -1
        elif self.data["position"].iloc[-1] == 0:
            if self.position == -1:
                order = self.create_order(self.instrument, self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
            elif self.position == 1:
                order = self.create_order(self.instrument, -self.units, suppress=True, ret=True)
                self.report_trade(order, "GOING NEUTRAL")
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


if __name__ == "__main__":
    lm = pickle.load(open("logreg.pkl", "rb"))
    trader = MLTrader("oanda.cfg", "EUR_USD", "5min", lags=4, model=lm, units=100000)
    trader.get_most_recent()
    trader.stream_data(trader.instrument, stop=500)
    if trader.position != 0:  # if we have a final open position
        close_order = trader.create_order(trader.instrument, units=-trader.position * trader.units,
                                          suppress=True, ret=True)
        trader.report_trade(close_order, "GOING NEUTRAL")
        trader.position = 0
