import pandas as pd
import numpy as np

train_corr = pd.read_csv("new_dataset/train_correlation.csv")
train_fund = pd.read_csv("new_dataset/train_fund_return.csv")
train_benchmark = pd.read_csv("new_dataset/train_fund_benchmark_return.csv")
train_index = pd.read_csv("new_dataset/train_index_return.csv", encoding="gbk")

train_index.columns = ["index_name"] + train_index.columns[1:].tolist()
train_benchmark.columns = ["fund_name"] + train_benchmark.columns[1:].tolist()
train_fund.columns = ["fund_name"] + train_fund.columns[1:].tolist()
train_corr.columns = ["fund_name"] + train_corr.columns[1:].tolist()

test_corr = pd.read_csv("new_dataset/test_correlation.csv")
test_fund = pd.read_csv("new_dataset/test_fund_return.csv")
test_benchmark = pd.read_csv("new_dataset/test_fund_benchmark_return.csv")
test_index = pd.read_csv("new_dataset/test_index_return.csv", encoding="gbk")

test_index.columns = ["index_name"] + test_index.columns[1:].tolist()
test_benchmark.columns = ["fund_name"] + test_benchmark.columns[1:].tolist()
test_fund.columns = ["fund_name"] + test_fund.columns[1:].tolist()
test_corr.columns = ["fund_name"] + test_corr.columns[1:].tolist()

fund_data = pd.merge(train_fund, test_fund, on="fund_name")
bench_data = pd.merge(train_benchmark, test_benchmark, on="fund_name")
index_data = pd.merge(train_index, test_index, on="index_name")
corr_data = pd.merge(train_corr, test_corr, on="fund_name")

fund_data.to_csv("new_dataset/fund_data.csv", index=None)
bench_data.to_csv("new_dataset/bench_data.csv", index=None)
index_data.to_csv("new_dataset/index_data.csv", index=None, encoding="gbk")
corr_data.to_csv("new_dataset/corr_data.csv", index=None)


