import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_next_day(curr_day):
    idx = all_date.index(curr_day)
    return all_date[idx+1]


def get_feat(start, w_size, have_label=True):
    end = start + w_size
    cols = [e for e in range(start, end)]
    fund_tmp = fund_data.iloc[:, cols]
    bench_tmp = bench_data.iloc[:, cols]
    index_tmp = index_data.iloc[:, cols]

    t = fund_data[["fund_name"]]
    t["fund_max"] = fund_tmp.max(axis=1)
    t["fund_min"] = fund_tmp.min(axis=1)
    t["fund_mean"] = fund_tmp.mean(axis=1)
    t["fund_std"] = fund_tmp.std(axis=1)

    t["fund_max_7"] = fund_tmp.iloc[:, -7:].max(axis=1)
    t["fund_min_7"] = fund_tmp.iloc[:, -7:].min(axis=1)
    t["fund_mean_7"] = fund_tmp.iloc[:, -7:].mean(axis=1)
    t["fund_std_7"] = fund_tmp.iloc[:, -7:].std(axis=1)

    t["fund_max_15"] = fund_tmp.iloc[:, -15:].max(axis=1)
    t["fund_min_15"] = fund_tmp.iloc[:, -15:].min(axis=1)
    t["fund_mean_15"] = fund_tmp.iloc[:, -15:].mean(axis=1)
    t["fund_std_15"] = fund_tmp.iloc[:, -15:].std(axis=1)

    t["fund_max_30"] = fund_tmp.iloc[:, -30:].max(axis=1)
    t["fund_min_30"] = fund_tmp.iloc[:, -30:].min(axis=1)
    t["fund_mean_30"] = fund_tmp.iloc[:, -30:].mean(axis=1)
    t["fund_std_30"] = fund_tmp.iloc[:, -30:].std(axis=1)

    fund_diff= fund_tmp.diff(axis=1).iloc[:, 1:]
    t["fund_diff_max"] = np.max(fund_diff, axis=1)
    t["fund_diff_min"] = np.min(fund_diff, axis=1)
    t["fund_diff_mean"] = np.mean(fund_diff, axis=1)
    t["fund_diff_std"] = np.std(fund_diff, axis=1)

    t["bench_max"] = bench_tmp.max(axis=1)
    t["bench_min"] = bench_tmp.min(axis=1)
    t["bench_mean"] = bench_tmp.mean(axis=1)
    t["bench_std"] = bench_tmp.std(axis=1)

    t["bench_max_7"] = bench_tmp.iloc[:, -7:].max(axis=1)
    t["bench_min_7"] = bench_tmp.iloc[:, -7:].min(axis=1)
    t["bench_mean_7"] = bench_tmp.iloc[:, -7:].mean(axis=1)
    t["bench_std_7"] = bench_tmp.iloc[:, -7:].std(axis=1)

    t["bench_max_15"] = bench_tmp.iloc[:, -15:].max(axis=1)
    t["bench_min_15"] = bench_tmp.iloc[:, -15:].min(axis=1)
    t["bench_mean_15"] = bench_tmp.iloc[:, -15:].mean(axis=1)
    t["bench_std_15"] = bench_tmp.iloc[:, -15:].std(axis=1)

    t["bench_max_30"] = bench_tmp.iloc[:, -30:].max(axis=1)
    t["bench_min_30"] = bench_tmp.iloc[:, -30:].min(axis=1)
    t["bench_mean_30"] = bench_tmp.iloc[:, -30:].mean(axis=1)
    t["bench_std_30"] = bench_tmp.iloc[:, -30:].std(axis=1)

    bench_diff = bench_tmp.diff(axis=1).iloc[:, 1:]
    t["benchmark_diff_max"] = np.max(bench_diff, axis=1)
    t["benchmark_diff_min"] = np.min(bench_diff, axis=1)
    t["benchmark_diff_mean"] = np.mean(bench_diff, axis=1)
    t["benchmark_diff_std"] = np.std(bench_diff, axis=1)

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

    fund_corr = fund_tmp.T.corr().values[decare_iindex]
    d1["fund_pearson"] = fund_corr

    fund_corr_7 = fund_tmp.iloc[:, -7:].T.corr().values[decare_iindex]
    d1["fund_pearson_7"] = fund_corr_7

    fund_corr_15 = fund_tmp.iloc[:, -15:].T.corr().values[decare_iindex]
    d1["fund_pearson_15"] = fund_corr_15

    fund_corr_30 = fund_tmp.iloc[:, -30:].T.corr().values[decare_iindex]
    d1["fund_pearson_30"] = fund_corr_30

    bench_corr = bench_tmp.T.corr().values[decare_iindex]
    d1["bench_pearson"] = bench_corr

    bench_corr_7 = bench_tmp.iloc[:, -7:].T.corr().values[decare_iindex]
    d1["bench_pearson_7"] = bench_corr_7

    bench_corr_15 = bench_tmp.iloc[:, -15:].T.corr().values[decare_iindex]
    d1["bench_pearson_15"] = bench_corr_15

    bench_corr_30 = bench_tmp.iloc[:, -30:].T.corr().values[decare_iindex]
    d1["bench_pearson_30"] = bench_corr_30

    fund_diff_corr = fund_diff.T.corr().values[decare_iindex]
    d1["fund_diff_pearson"] = fund_diff_corr

    fund_diff_corr_7 = fund_diff.iloc[:, -7:].T.corr().values[decare_iindex]
    d1["fund_diff_pearson_7"] = fund_diff_corr_7

    fund_diff_corr_15 = fund_diff.iloc[:, -15:].T.corr().values[decare_iindex]
    d1["fund_diff_pearson_15"] = fund_diff_corr_15

    fund_diff_corr_30 = fund_diff.iloc[:, -30:].T.corr().values[decare_iindex]
    d1["fund_diff_pearson_30"] = fund_diff_corr_30

    bench_diff_corr = bench_diff.T.corr().values[decare_iindex]
    d1["bench_diff_pearson"] = bench_diff_corr

    bench_diff_corr_7 = bench_diff.iloc[:, -7:].T.corr().values[decare_iindex]
    d1["bench_diff_pearson_7"] = bench_diff_corr_7

    bench_diff_corr_15 = bench_diff.iloc[:, -15:].T.corr().values[decare_iindex]
    d1["bench_diff_pearson_15"] = bench_diff_corr_15

    bench_diff_corr_30 = bench_diff.iloc[:, -30:].T.corr().values[decare_iindex]
    d1["bench_diff_pearson_30"] = bench_diff_corr_30

    fund_corr = fund_tmp.T.corr("spearman").values[decare_iindex]
    d1["fund_spearman"] = fund_corr

    fund_corr_7 = fund_tmp.iloc[:, -7:].T.corr("spearman").values[decare_iindex]
    d1["fund_spearman_7"] = fund_corr_7

    fund_corr_15 = fund_tmp.iloc[:, -15:].T.corr("spearman").values[decare_iindex]
    d1["fund_spearman_15"] = fund_corr_15

    fund_corr_30 = fund_tmp.iloc[:, -30:].T.corr("spearman").values[decare_iindex]
    d1["fund_spearman_30"] = fund_corr_30

    bench_corr = bench_tmp.T.corr("spearman").values[decare_iindex]
    d1["bench_spearman"] = bench_corr

    bench_corr_7 = bench_tmp.iloc[:, -7:].T.corr("spearman").values[decare_iindex]
    d1["bench_spearman_7"] = bench_corr_7

    bench_corr_15 = bench_tmp.iloc[:, -15:].T.corr("spearman").values[decare_iindex]
    d1["bench_spearman_15"] = bench_corr_15

    bench_corr_30 = bench_tmp.iloc[:, -30:].T.corr("spearman").values[decare_iindex]
    d1["bench_spearman_30"] = bench_corr_30

    fund_diff_corr = fund_diff.T.corr("spearman").values[decare_iindex]
    d1["fund_diff_spearman"] = fund_diff_corr

    fund_diff_corr_7 = fund_diff.iloc[:, -7:].T.corr("spearman").values[decare_iindex]
    d1["fund_diff_spearman_7"] = fund_diff_corr_7

    fund_diff_corr_15 = fund_diff.iloc[:, -15:].T.corr("spearman").values[decare_iindex]
    d1["fund_diff_spearman_15"] = fund_diff_corr_15

    fund_diff_corr_30 = fund_diff.iloc[:, -30:].T.corr("spearman").values[decare_iindex]
    d1["fund_diff_spearman_30"] = fund_diff_corr_30

    bench_diff_corr = bench_diff.T.corr("spearman").values[decare_iindex]
    d1["bench_diff_spearman"] = bench_diff_corr

    bench_diff_corr_7 = bench_diff.iloc[:, -7:].T.corr("spearman").values[decare_iindex]
    d1["bench_diff_spearman_7"] = bench_diff_corr_7

    bench_diff_corr_15 = bench_diff.iloc[:, -15:].T.corr("spearman").values[decare_iindex]
    d1["bench_diff_spearman_15"] = bench_diff_corr_15

    bench_diff_corr_30 = bench_diff.iloc[:, -30:].T.corr("spearman").values[decare_iindex]
    d1["bench_diff_spearman_30"] = bench_diff_corr_30

    fund_corr = cosine_similarity(fund_tmp).values[decare_iindex]
    d1["fund_cosin"] = fund_corr

    fund_corr_7 = cosine_similarity(fund_tmp.iloc[:, -7:]).values[decare_iindex]
    d1["fund_cosin_7"] = fund_corr_7

    fund_corr_15 = cosine_similarity(fund_tmp.iloc[:, -15:]).alues[decare_iindex]
    d1["fund_cosin_15"] = fund_corr_15

    fund_corr_30 = cosine_similarity(fund_tmp.iloc[:, -30:]).values[decare_iindex]
    d1["fund_cosin_30"] = fund_corr_30

    bench_corr = cosine_similarity(bench_tmp).values[decare_iindex]
    d1["bench_cosin"] = bench_corr

    bench_corr_7 = cosine_similarity(bench_tmp.iloc[:, -7:]).values[decare_iindex]
    d1["bench_cosin_7"] = bench_corr_7

    bench_corr_15 = cosine_similarity(bench_tmp.iloc[:, -15:]).values[decare_iindex]
    d1["bench_cosin_15"] = bench_corr_15

    bench_corr_30 = cosine_similarity(bench_tmp.iloc[:, -30:]).values[decare_iindex]
    d1["bench_cosin_30"] = bench_corr_30

    fund_diff_corr = cosine_similarity(fund_diff).values[decare_iindex]
    d1["fund_diff_cosin"] = fund_diff_corr

    fund_diff_corr_7 = cosine_similarity(fund_diff.iloc[:, -7:]).values[decare_iindex]
    d1["fund_diff_cosin_7"] = fund_diff_corr_7

    fund_diff_corr_15 = cosine_similarity(fund_diff.iloc[:, -15:]).values[decare_iindex]
    d1["fund_diff_cosin_15"] = fund_diff_corr_15

    fund_diff_corr_30 = cosine_similarity(fund_diff.iloc[:, -30:]).values[decare_iindex]
    d1["fund_diff_cosin_30"] = fund_diff_corr_30

    bench_diff_corr = cosine_similarity(bench_diff).values[decare_iindex]
    d1["bench_diff_cosin"] = bench_diff_corr

    bench_diff_corr_7 = cosine_similarity(bench_diff.iloc[:, -7:]).values[decare_iindex]
    d1["bench_diff_cosin_7"] = bench_diff_corr_7

    bench_diff_corr_15 = cosine_similarity(bench_diff.iloc[:, -15:]).values[decare_iindex]
    d1["bench_diff_cosin_15"] = bench_diff_corr_15

    bench_diff_corr_30 = cosine_similarity(bench_diff.iloc[:, -30:]).values[decare_iindex]
    d1["bench_diff_cosin_30"] = bench_diff_corr_30


    # 相关性特征
    corr_value = corr_data.iloc[:, start-15:start]
    d1["corr_max_15"] = corr_value.max(axis=1)
    d1["corr_min_15"] = corr_value.min(axis=1)
    d1["corr_mean_15"] = corr_value.mean(axis=1)
    d1["corr_std_15"] = corr_value.std(axis=1)

    corr_value = corr_data.iloc[:, start-7:start]
    d1["corr_max_7"] = corr_value.max(axis=1)
    d1["corr_min_7"] = corr_value.min(axis=1)
    d1["corr_mean_7"] = corr_value.mean(axis=1)
    d1["corr_std_7"] = corr_value.std(axis=1)

    if have_label:
        next_day = get_next_day(index_tmp.columns[-1])
        d1["label"] = corr_data[next_day]
        print(next_day)

    return d1


if __name__ == '__main__':
    fund_data = pd.read_csv("dataset/fund_data.csv")
    bench_data = pd.read_csv("dataset/bench_data.csv")
    index_data = pd.read_csv("dataset/index_data.csv", encoding="gbk")
    corr_data = pd.read_csv("dataset/corr_data.csv")
    all_date = corr_data.columns[1:].tolist()

    w_size = 60 # 窗口大小
    # delta = 30 # 步长
    # data_list = []
    # for i in range(1, 481, delta):
    #     data = get_feat(i, w_size).iloc[:, 1:]
    #     data_list.append(data)

    # train_data = pd.concat(data_list, axis=0)
    # print(train_data.shape)
    # train_data.to_csv("new_data/train_data1_481_30.csv", index=None)

    test_data = get_feat(541, w_size, have_label=False)
    # test_data.to_csv("data/test_data.csv", index=None)