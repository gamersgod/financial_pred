import pandas as pd
import numpy as np


def get_dataset(i_data, w_size, time_delta, index_data, fund_data, benchmark_data, corr_data):
    start = i_data * time_delta + 1
    cols = [0] + [e for e in range(start, start+w_size)]
    print(cols)
    index_tmp = index_data.iloc[:, cols]
    fund_tmp = fund_data.iloc[:, cols]
    benchmark_tmp = benchmark_data.iloc[:, cols]
    
    t = fund_tmp[["fund_name"]]
    t["fund_max"] = fund_tmp.max(axis=1)
    t["fund_min"] = fund_tmp.min(axis=1)
    t["fund_mean"] = fund_tmp.mean(axis=1)
    t["fund_std"] = fund_tmp.std(axis=1)
    
    t["benchmark_max"] = benchmark_tmp.max(axis=1)
    t["benchmark_min"] = benchmark_tmp.min(axis=1)
    t["benchmark_mean"] = benchmark_tmp.mean(axis=1)
    t["benchmark_std"] = benchmark_tmp.std(axis=1)
    
    sub_tmp = fund_tmp.iloc[:, 1:].sub(benchmark_tmp.iloc[:, 1:])
    t["sub_max"] = sub_tmp.max(axis=1)
    t["sub_min"] = sub_tmp.min(axis=1)
    t["sub_mean"] = sub_tmp.mean(axis=1)
    t["sub_std"] = sub_tmp.std(axis=1)
    
    div0_tmp = fund_tmp.iloc[:, 1:].div(benchmark_tmp.iloc[:, 1:])
    t["div0_max"] = div0_tmp.max(axis=1)
    t["div0_min"] = div0_tmp.min(axis=1)
    t["div0_mean"] = div0_tmp.mean(axis=1)
    t["div0_std"] = div0_tmp.std(axis=1)
    
    div1_tmp = benchmark_tmp.iloc[:, 1:].div(fund_tmp.iloc[:, 1:])
    t["div1_max"] = div1_tmp.max(axis=1)
    t["div1_min"] = div1_tmp.min(axis=1)
    t["div1_mean"] = div1_tmp.mean(axis=1)
    t["div1_std"] = div1_tmp.std(axis=1)
    
    t["fb_max_diff"] = t["fund_max"] - t["benchmark_max"]
    t["fb_min_diff"] = t["fund_min"] - t["benchmark_min"]
    t["fb_mean_diff"] = t["fund_mean"] - t["benchmark_mean"]
    t["fb_std_diff"] = t["fund_std"] - t["benchmark_std"]
    t["fb_max_rate"] = t["fund_max"] / t["benchmark_max"]
    t["fb_min_rate"] = t["fund_min"] / t["benchmark_min"]
    t["fb_mean_rate"] = t["fund_mean"] / t["benchmark_mean"]
    t["fb_std_rate"] = t["fund_std"] / t["benchmark_std"]
    
    columns = t.columns.tolist() + [i+"_x" for i in t.columns[1:].tolist()]
    t1 = pd.DataFrame(columns=columns)
    
    fund_names = t["fund_name"].tolist()
    index = 0
    for i in range(len(fund_names)):
        for j in range(i+1, len(fund_names)):
            d = [fund_names[i]+"-"+fund_names[j]] + t.iloc[i, 1:].append(t.iloc[j, 1:], ignore_index=True).tolist()
            t1.loc[index] = d
            index += 1
    
    feats = t.columns[1:]
    i = 0
    for feat in feats:
        t1["new_feat_%s"%(i)] = t1[feat] - t1[feat+"_x"]
        t1["new_feat_%s"%(i+1)] = t1[feat] / t1[feat+"_x"]
        i += 2
    
    all_date = corr_data.columns[1:].tolist()
    next_date = all_date[all_date.index(index_tmp.columns[-1])+ 1]
    t1["label"] = corr_data[next_date]
    
    t1.to_csv("data/train_data_{}.csv".format(i_data), index=None)
    

if __name__ == '__main__':
    train_corr = pd.read_csv("dataset/train_correlation.csv")
    train_fund = pd.read_csv("dataset/train_fund_return.csv")
    train_benchmark = pd.read_csv("dataset/train_fund_benchmark_return.csv")
    train_index = pd.read_csv("dataset/train_index_return.csv", encoding="gbk")

    train_index.columns = ["index_name"] + train_index.columns[1:].tolist()
    train_benchmark.columns = ["fund_name"] + train_benchmark.columns[1:].tolist()
    train_fund.columns = ["fund_name"] + train_fund.columns[1:].tolist()

    time_delta = 30 # 步长
    w_size = 30  # 窗口大小
    for k in range(0, 11):
        get_dataset(k, w_size, time_delta, train_index, train_fund, train_benchmark, train_corr)
        print("done!")
