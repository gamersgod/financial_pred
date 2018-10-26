import pandas as pd
import numpy as np


def get_eluer_dis(x, y):
    return np.linalg.norm(x - y)


def get_cosin_dis(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def get_next_day(curr_day):
    idx = all_date.index(curr_day)
    return all_date[idx+1]


def get_feat(start, w_size, have_label=True):
    cols = [e for e in range(start, start+w_size)]
    fund_names = fund_data["fund_name"].tolist()
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

    fund_diff_value = np.diff(fund_tmp.values)
    t["fund_diff_max"] = np.max(fund_diff_value, axis=1)
    t["fund_diff_min"] = np.min(fund_diff_value, axis=1)
    t["fund_diff_mean"] = np.mean(fund_diff_value, axis=1)
    t["fund_diff_std"] = np.std(fund_diff_value, axis=1)

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

    bench_diff_value = np.diff(bench_tmp.values)
    t["benchmark_diff_max"] = np.max(bench_diff_value, axis=1)
    t["benchmark_diff_min"] = np.min(bench_diff_value, axis=1)
    t["benchmark_diff_mean"] = np.mean(bench_diff_value, axis=1)
    t["benchmark_diff_std"] = np.std(bench_diff_value, axis=1)

    # sub_tmp = fund_tmp.sub(bench_tmp)
    # t["sub_max"] = sub_tmp.max(axis=1)
    # t["sub_min"] = sub_tmp.min(axis=1)
    # t["sub_mean"] = sub_tmp.mean(axis=1)
    # t["sub_std"] = sub_tmp.std(axis=1)
    #
    # t["sub_max_7"] = sub_tmp.iloc[:, -7:].max(axis=1)
    # t["sub_min_7"] = sub_tmp.iloc[:, -7:].min(axis=1)
    # t["sub_mean_7"] = sub_tmp.iloc[:, -7:].mean(axis=1)
    # t["sub_std_7"] = sub_tmp.iloc[:, -7:].std(axis=1)
    #
    # t["sub_max_15"] = sub_tmp.iloc[:, -15:].max(axis=1)
    # t["sub_min_15"] = sub_tmp.iloc[:, -15:].min(axis=1)
    # t["sub_mean_15"] = sub_tmp.iloc[:, -15:].mean(axis=1)
    # t["sub_std_15"] = sub_tmp.iloc[:, -15:].std(axis=1)
    #
    # t["sub_max_30"] = sub_tmp.iloc[:, -30:].max(axis=1)
    # t["sub_min_30"] = sub_tmp.iloc[:, -30:].min(axis=1)
    # t["sub_mean_30"] = sub_tmp.iloc[:, -30:].mean(axis=1)
    # t["sub_std_30"] = sub_tmp.iloc[:, -30:].std(axis=1)
    #
    # div0_tmp = fund_tmp.iloc.div(bench_tmp)
    # t["div0_max"] = div0_tmp.max(axis=1)
    # t["div0_min"] = div0_tmp.min(axis=1)
    # t["div0_mean"] = div0_tmp.mean(axis=1)
    # t["div0_std"] = div0_tmp.std(axis=1)
    #
    # div1_tmp = bench_tmp.div(fund_tmp)
    # t["div1_max"] = div1_tmp.max(axis=1)
    # t["div1_min"] = div1_tmp.min(axis=1)
    # t["div1_mean"] = div1_tmp.mean(axis=1)
    # t["div1_std"] = div1_tmp.std(axis=1)
    #
    # t["fb_max_diff"] = t["fund_max"] - t["benchmark_max"]
    # t["fb_min_diff"] = t["fund_min"] - t["benchmark_min"]
    # t["fb_mean_diff"] = t["fund_mean"] - t["benchmark_mean"]
    # t["fb_std_diff"] = t["fund_std"] - t["benchmark_std"]
    #
    # t["fb_diff_max_diff"] = t["fund_diff_max"] - t["benchmark_diff_max"]
    # t["fb_diff_min_diff"] = t["fund_diff_min"] - t["benchmark_diff_min"]
    # t["fb_diff_mean_diff"] = t["fund_diff_mean"] - t["benchmark_diff_mean"]
    # t["fb_diff_std_diff"] = t["fund_diff_std"] - t["benchmark_diff_std"]
    #
    # t["fb_max_rate"] = t["fund_max"] / t["benchmark_max"]
    # t["fb_min_rate"] = t["fund_min"] / t["benchmark_min"]
    # t["fb_mean_rate"] = t["fund_mean"] / t["benchmark_mean"]
    # t["fb_std_rate"] = t["fund_std"] / t["benchmark_std"]
    #
    # t["fb_diff_max_rate"] = t["fund_diff_max"] / t["benchmark_diff_max"]
    # t["fb_diff_min_rate"] = t["fund_diff_min"] / t["benchmark_diff_min"]
    # t["fb_diff_mean_rate"] = t["fund_diff_mean"] / t["benchmark_diff_mean"]
    # t["fb_diff_std_rate"] = t["fund_diff_std"] / t["benchmark_diff_std"]

    columns = t.columns.tolist() + [i+"_x" for i in t.columns[1:].tolist()] + \
                                                                               ["fund_eluer",
                                                                               "fund_cosin",
                                                                               "bench_eluer",
                                                                               "bench_cosin",
                                                                               "fund_diff_eluer",
                                                                               "fund_diff_cosin",
                                                                               "bench_diff_eluer",
                                                                               "bench_diff_cosin",
                                                                               "fund_eluer_7",
                                                                               "fund_eluer_15",
                                                                               "fund_eluer_30",
                                                                               "fund_cosin_7",
                                                                               "fund_cosin_15",
                                                                               "fund_cosin_30",
                                                                               "bench_eluer_7",
                                                                               "bench_eluer_15",
                                                                               "bench_eluer_30",
                                                                               "bench_cosin_7",
                                                                               "bench_cosin_15",
                                                                               "bench_cosin_30",
                                                                               "fund_diff_eluer_7",
                                                                               "fund_diff_eluer_15",
                                                                               "fund_diff_eluer_30",
                                                                               "fund_diff_cosin_7",
                                                                               "fund_diff_cosin_15",
                                                                               "fund_diff_cosin_30",
                                                                               "bench_diff_eluer_7",
                                                                               "bench_diff_eluer_15",
                                                                               "bench_diff_eluer_30",
                                                                               "bench_diff_cosin_7",
                                                                               "bench_diff_cosin_15",
                                                                               "bench_diff_cosin_30"]

    t1 = pd.DataFrame(columns=columns)
    index = 0
    for i in range(len(fund_names)):
        for j in range(i + 1, len(fund_names)):
            d = [fund_names[i] + "-" + fund_names[j]] + t.iloc[i, 1:].append(t.iloc[j, 1:], ignore_index=True).tolist()

            fund_x, fund_y = fund_tmp.iloc[i, :].values, fund_tmp.iloc[j, :].values
            fund_eluer = get_eluer_dis(fund_x, fund_y)
            fund_cosin = get_cosin_dis(fund_x, fund_y)

            fund_x_7, fund_y_7 = fund_tmp.iloc[i, -7:].values, fund_tmp.iloc[j, -7:].values
            fund_eluer_7 = get_eluer_dis(fund_x_7, fund_y_7)
            fund_cosin_7 = get_cosin_dis(fund_x_7, fund_y_7)

            fund_x_15, fund_y_15 = fund_tmp.iloc[i, -15:].values, fund_tmp.iloc[j, -15:].values
            fund_eluer_15 = get_eluer_dis(fund_x_15, fund_y_15)
            fund_cosin_15 = get_cosin_dis(fund_x_15, fund_y_15)

            fund_x_30, fund_y_30 = fund_tmp.iloc[i, -30:].values, fund_tmp.iloc[j, -30:].values
            fund_eluer_30 = get_eluer_dis(fund_x_30, fund_y_30)
            fund_cosin_30 = get_cosin_dis(fund_x_30, fund_y_30)

            bench_x, bench_y = bench_tmp.iloc[i, :].values, bench_tmp.iloc[j, :].values
            bench_eluer = get_eluer_dis(bench_x, bench_y)
            bench_cosin = get_cosin_dis(bench_x, bench_y)

            bench_x_7, bench_y_7 = bench_tmp.iloc[i, -7:].values, bench_tmp.iloc[j, -7:].values
            bench_eluer_7 = get_eluer_dis(bench_x_7, bench_y_7)
            bench_cosin_7 = get_cosin_dis(bench_x_7, bench_y_7)

            bench_x_15, bench_y_15 = bench_tmp.iloc[i, -15:].values, bench_tmp.iloc[j, -15:].values
            bench_eluer_15 = get_eluer_dis(bench_x_15, bench_y_15)
            bench_cosin_15 = get_cosin_dis(bench_x_15, bench_y_15)

            bench_x_30, bench_y_30 = bench_tmp.iloc[i, -30:].values, bench_tmp.iloc[j, -30:].values
            bench_eluer_30 = get_eluer_dis(bench_x_30, bench_y_30)
            bench_cosin_30 = get_cosin_dis(bench_x_30, bench_y_30)

            fund_diff_x, fund_diff_y = fund_diff_value[i, :], fund_diff_value[j, :]
            fund_diff_eluer = get_eluer_dis(fund_diff_x, fund_diff_y)
            fund_diff_cosin = get_cosin_dis(fund_diff_x, fund_diff_y)

            fund_diff_x_7, fund_diff_y_7 = fund_diff_value[i, -7:], fund_diff_value[j, -7:]
            fund_diff_eluer_7 = get_eluer_dis(fund_diff_x_7, fund_diff_y_7)
            fund_diff_cosin_7 = get_cosin_dis(fund_diff_x_7, fund_diff_y_7)

            fund_diff_x_15, fund_diff_y_15 = fund_diff_value[i, -15:], fund_diff_value[j, -15:]
            fund_diff_eluer_15 = get_eluer_dis(fund_diff_x_15, fund_diff_y_15)
            fund_diff_cosin_15 = get_cosin_dis(fund_diff_x_15, fund_diff_y_15)

            fund_diff_x_30, fund_diff_y_30 = fund_diff_value[i, -30:], fund_diff_value[j, -30:]
            fund_diff_eluer_30 = get_eluer_dis(fund_diff_x_30, fund_diff_y_30)
            fund_diff_cosin_30 = get_cosin_dis(fund_diff_x_30, fund_diff_y_30)

            bench_diff_x, bench_diff_y = bench_diff_value[i, :], bench_diff_value[j, :]
            bench_diff_eluer = get_eluer_dis(bench_diff_x, bench_diff_y)
            bench_diff_cosin = get_cosin_dis(bench_diff_x, bench_diff_y)

            bench_diff_x_7, bench_diff_y_7 = bench_diff_value[i, -7:], bench_diff_value[j, -7:]
            bench_diff_eluer_7 = get_eluer_dis(bench_diff_x_7, bench_diff_y_7)
            bench_diff_cosin_7 = get_cosin_dis(bench_diff_x_7, bench_diff_y_7)

            bench_diff_x_15, bench_diff_y_15 = bench_diff_value[i, -15:], bench_diff_value[j, -15:]
            bench_diff_eluer_15 = get_eluer_dis(bench_diff_x_15, bench_diff_y_15)
            bench_diff_cosin_15 = get_cosin_dis(bench_diff_x_15, bench_diff_y_15)

            bench_diff_x_30, bench_diff_y_30 = bench_diff_value[i, -30:], bench_diff_value[j, -30:]
            bench_diff_eluer_30 = get_eluer_dis(bench_diff_x_30, bench_diff_y_30)
            bench_diff_cosin_30 = get_cosin_dis(bench_diff_x_30, bench_diff_y_30)

            t1.loc[index] = d + [fund_eluer, fund_cosin, bench_eluer, bench_cosin, fund_diff_eluer,
                                 fund_diff_cosin, bench_diff_eluer, bench_diff_cosin, fund_eluer_7,
                                 fund_eluer_15, fund_eluer_30, fund_cosin_7, fund_cosin_15, fund_cosin_30,
                                 bench_eluer_7, bench_eluer_15, bench_eluer_30, bench_cosin_7,
                                 bench_cosin_15, bench_cosin_30, fund_diff_eluer_7, fund_diff_eluer_15,
                                 fund_diff_eluer_30, fund_diff_cosin_7, fund_diff_cosin_15,
                                 fund_diff_cosin_30, bench_diff_eluer_7, bench_diff_eluer_15,
                                 bench_diff_eluer_30, bench_diff_cosin_7, bench_diff_cosin_15,
                                 bench_diff_cosin_30]
            index += 1


    if have_label:
        next_day = get_next_day(index_tmp.columns[-1])
        t1["label"] = corr_data[next_day]
        print(next_day)

    return t1


if __name__ == '__main__':
    fund_data = pd.read_csv("new_dataset/fund_data.csv")
    bench_data = pd.read_csv("new_dataset/bench_data.csv")
    index_data = pd.read_csv("new_dataset/index_data.csv", encoding="gbk")
    corr_data = pd.read_csv("new_dataset/corr_data.csv")
    all_date = corr_data.columns[1:].tolist()

    # TODO: 两个基金之前的相关性特征，只能取前多少天的，但是预测的时候有60多天corr数据缺失，怎么办
    w_size = 60 # 窗口大小
    delta = 30 # 步长
    data_list = []
    for i in range(300, 481, delta):
        data = get_feat(i, w_size).iloc[:, 1:]
        data_list.append(data)

    train_data = pd.concat(data_list, axis=0)
    print(train_data.shape)
    train_data.to_csv("new_data/train_data300_481_30.csv", index=None)

    # test_data = get_feat(541, 60, have_label=False)
    # test_data.to_csv("data/test_data.csv", index=None)