import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


def get_next_day(curr_day):
    idx = all_date.index(curr_day)
    return all_date[idx+1]


def get_feat(start, w_size, have_label=True):
    end = start + w_size
    cols = [e for e in range(start, end)]
    fund_tmp = fund_data.iloc[:, cols]
    bench_tmp = bench_data.iloc[:, cols]
    index_tmp = index_data.iloc[:, cols]
    fund_diff= fund_tmp.diff(axis=1).iloc[:, 1:]
    bench_diff = bench_tmp.diff(axis=1).iloc[:, 1:]

    fund_tmp_7 = fund_tmp.iloc[:, -7:]
    bench_tmp_7 = bench_tmp.iloc[:, -7:]
    fund_diff_7 = fund_diff.iloc[:, -7:]
    bench_diff_7 = bench_diff.iloc[:, -7:]

    t = fund_data[["fund_name"]]
    t["fund_max"] = fund_tmp.max(axis=1)
    t["fund_min"] = fund_tmp.min(axis=1)
    t["fund_mean"] = fund_tmp.mean(axis=1)
    t["fund_std"] = fund_tmp.std(axis=1)

    t["fund_max_7"] = fund_tmp_7.max(axis=1)
    t["fund_min_7"] = fund_tmp_7.min(axis=1)
    t["fund_mean_7"] = fund_tmp_7.mean(axis=1)
    t["fund_std_7"] = fund_tmp_7.std(axis=1)

    t["fund_diff_max"] = np.max(fund_diff, axis=1)
    t["fund_diff_min"] = np.min(fund_diff, axis=1)
    t["fund_diff_mean"] = np.mean(fund_diff, axis=1)
    t["fund_diff_std"] = np.std(fund_diff, axis=1)

    t["bench_max"] = bench_tmp.max(axis=1)
    t["bench_min"] = bench_tmp.min(axis=1)
    t["bench_mean"] = bench_tmp.mean(axis=1)
    t["bench_std"] = bench_tmp.std(axis=1)

    t["bench_max_7"] = bench_tmp_7.max(axis=1)
    t["bench_min_7"] = bench_tmp_7.min(axis=1)
    t["bench_mean_7"] = bench_tmp_7.mean(axis=1)
    t["bench_std_7"] = bench_tmp_7.std(axis=1)

    t["bench_diff_max"] = np.max(bench_diff, axis=1)
    t["bench_diff_min"] = np.min(bench_diff, axis=1)
    t["bench_diff_mean"] = np.mean(bench_diff, axis=1)
    t["bench_diff_std"] = np.std(bench_diff, axis=1)

    t["value"] = 1
    d = pd.merge(t, t, how="inner", on="value")
    d["fund_name_x_"] = d["fund_name_x"].apply(lambda i: int(i.split()[-1]))
    d["fund_name_y_"] = d["fund_name_y"].apply(lambda i: int(i.split()[-1]))
    d = d[d["fund_name_y_"] > d["fund_name_x_"]]

    x_index = d["fund_name_x_"].values - 1
    y_index = d["fund_name_y_"].values - 1
    decare_iindex = [x_index, y_index]

    d["fund_name"] = d["fund_name_x"] + "-" + d["fund_name_y"]
    d1 = d.drop(["fund_name_x", "fund_name_y", "fund_name_x_", "fund_name_y_", "value"], axis=1)

    def get_pandas_corr(corr_type):
        fund_corr = fund_tmp.T.corr(corr_type).values[decare_iindex]
        d1["fund_" + corr_type] = fund_corr

        fund_corr_7 = fund_tmp_7.T.corr(corr_type).values[decare_iindex]
        d1["fund_" + corr_type + "_7"] = fund_corr_7

        bench_corr = bench_tmp.T.corr(corr_type).values[decare_iindex]
        d1["bench_" + corr_type] = bench_corr

        bench_corr_7 = bench_tmp_7.T.corr(corr_type).values[decare_iindex]
        d1["bench_" + corr_type + "_7"] = bench_corr_7

        fund_diff_corr = fund_diff.T.corr(corr_type).values[decare_iindex]
        d1["fund_diff_" + corr_type] = fund_diff_corr

        fund_diff_corr_7 = fund_diff_7.T.corr(corr_type).values[decare_iindex]
        d1["fund_diff_" + corr_type + "_7"] = fund_diff_corr_7

        bench_diff_corr = bench_diff.T.corr(corr_type).values[decare_iindex]
        d1["bench_diff_" + corr_type] = bench_diff_corr

        bench_diff_corr_7 = bench_diff_7.T.corr(corr_type).values[decare_iindex]
        d1["bench_diff_" + corr_type + "_7"] = bench_diff_corr_7

    def get_sklearn_dist(metric):
        fund_corr = cdist(fund_tmp, fund_tmp, metric=metric)[decare_iindex]
        d1["fund_" + metric] = fund_corr

        fund_corr_7 = cdist(fund_tmp_7,fund_tmp_7, metric=metric)[decare_iindex]
        d1["fund_" + metric + "_7"] = fund_corr_7

        bench_corr = cdist(bench_tmp, bench_tmp, metric=metric)[decare_iindex]
        d1["bench_" + metric] = bench_corr

        bench_corr_7 = cdist(bench_tmp_7, bench_tmp_7, metric=metric)[decare_iindex]
        d1["bench_" + metric + "_7"] = bench_corr_7

        fund_diff_corr = cdist(fund_diff, fund_diff, metric=metric)[decare_iindex]
        d1["fund_diff_" + metric] = fund_diff_corr

        fund_diff_corr_7 = cdist(fund_diff_7, fund_diff_7, metric=metric)[decare_iindex]
        d1["fund_diff_" + metric + "_7"] = fund_diff_corr_7

        bench_diff_corr = cdist(bench_diff, bench_diff, metric=metric)[decare_iindex]
        d1["bench_diff_" + metric] = bench_diff_corr

        bench_diff_corr_7 = cdist(bench_diff_7, bench_diff_7, metric=metric)[decare_iindex]
        d1["bench_diff_" + metric + "_7"] = bench_diff_corr_7

    get_pandas_corr("pearson")
    get_pandas_corr("spearman")
    # get_pandas_corr("kendall")
    get_sklearn_dist("cosine")
    # get_sklearn_dist("euclidean")
    # get_sklearn_dist("seuclidean")
    # get_sklearn_dist("mahalanobis")
    # get_sklearn_dist("braycurtis")
    # get_sklearn_dist("canberra")
    # get_sklearn_dist("chebyshev")


    # 相关性特征
    corr_value = corr_data.iloc[:, start-8:start-1]
    d1["corr_max_7"] = corr_value.max(axis=1).values
    d1["corr_min_7"] = corr_value.min(axis=1).values
    d1["corr_mean_7"] = corr_value.mean(axis=1).values
    d1["corr_std_7"] = corr_value.std(axis=1).values

    corr_value = corr_data.iloc[:, start-16:start-1]
    d1["corr_max_15"] = corr_value.max(axis=1).values
    d1["corr_min_15"] = corr_value.min(axis=1).values
    d1["corr_mean_15"] = corr_value.mean(axis=1).values
    d1["corr_std_15"] = corr_value.std(axis=1).values

    corr_value = corr_data.iloc[:, start-62:start-1]
    d1["corr_max"] = corr_value.max(axis=1).values
    d1["corr_min"] = corr_value.min(axis=1).values
    d1["corr_mean"] = corr_value.mean(axis=1).values
    d1["corr_std"] = corr_value.std(axis=1).values

    if have_label:
        next_day = get_next_day(index_tmp.columns[-1])
        d1["label"] = corr_data[next_day].values
        print(next_day)

    return d1


if __name__ == '__main__':
    dataset_dir = "dataset/"
    data_dir = "data/"
    fund_data = pd.read_csv(dataset_dir + "fund_data.csv")
    bench_data = pd.read_csv(dataset_dir + "bench_data.csv")
    index_data = pd.read_csv(dataset_dir + "index_data.csv", encoding="gbk")
    corr_data = pd.read_csv(dataset_dir + "corr_data.csv")
    all_date = corr_data.columns[1:].tolist()

    # 101-480
    w_size = 60 # 窗口大小
    delta = 5 # 步长
    data_list = []
    for i in range(100, 481, delta):
        data = get_feat(i, w_size)
        data_list.append(data)

    train_data = pd.concat(data_list, axis=0)
    print(train_data.shape)
    train_data.to_csv(data_dir + "train_data.csv", index=None)

    test_data = get_feat(541, w_size, have_label=False)
    test_data.to_csv(data_dir + "test_data.csv", index=None)