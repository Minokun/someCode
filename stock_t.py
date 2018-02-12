import tushare as ts
import time
if __name__ == '__main__':
    while (True):
        time.sleep(2)
        df = ts.get_realtime_quotes('600460')
        print(df[['code', 'name', 'price', 'bid', 'ask', 'volume', 'amount', 'time', 'b1_v', 'a1_v', 'b1_p', 'a1_p']].values)
