import csv
import pandas as pd
import tpqoa

if __name__ == "__main__":
    api = tpqoa.tpqoa("oanda.cfg")
    historicalData = api.get_history(instrument = "EUR_USD", start = "2017-01-02", end = "2022-07-24", granularity = "M30", price = "M")
    historicalData.to_csv("UpdateData.csv")