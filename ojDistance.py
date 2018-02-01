# 欧几里得距离
def caculate(df, i, n):
    res = []
    if n == len(df.index):
        return res
    else:
        sum_pow = 0
        for i_columns in df.columns:
            sum_pow += (df.loc[i][i_columns] - df.loc[n][i_columns]) ** 2
        res.append(np.sqrt(sum_pow))
        n += 1
        res = res + caculate(df, i, n)
        return res


def ojDistanse(df):
    list_distance = []
    for i in range(len(df.index) - 1):
        res_list = caculate(df, i, i + 1)
        list_distance.append(res_list)
    return list_distance


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.DataFrame([pd.Series(np.array([1.5, 2, 1.6, 1.2, 1.5])), np.array([1.7, 1.9, 1.8, 1.5, 1])]).T.rename(
        columns={0: 'A1', 1: 'A2'})
    print(ojDistanse(df))