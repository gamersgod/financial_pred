import pandas as pd
import lightgbm as lgb

train_data0 = pd.read_csv("data/train_data_0.csv").iloc[:, 50:]
train_data1 = pd.read_csv("data/train_data_1.csv").iloc[:, 50:]
train_data2 = pd.read_csv("data/train_data_2.csv").iloc[:, 50:]
train_data3 = pd.read_csv("data/train_data_3.csv").iloc[:, 50:]
train_data4 = pd.read_csv("data/train_data_4.csv").iloc[:, 50:]
train_data5 = pd.read_csv("data/train_data_5.csv").iloc[:, 50:]
train_data6 = pd.read_csv("data/train_data_6.csv").iloc[:, 50:]
train_data7 = pd.read_csv("data/train_data_7.csv").iloc[:, 50:]
train_data8 = pd.read_csv("data/train_data_8.csv").iloc[:, 50:]
train_data9 = pd.read_csv("data/train_data_9.csv").iloc[:, 50:]
train_data10 = pd.read_csv("data/train_data_10.csv").iloc[:, 50:]

train_data = pd.concat([train_data0, train_data1, train_data2, train_data3, train_data4, train_data5, train_data6,
                        train_data7, train_data8], axis=0)
print(train_data.shape)

test_data = pd.concat([train_data9, train_data10], axis=0)
print(test_data.shape)

train_x = train_data.iloc[:, :-1]
train_y = train_data["label"]

test_x = test_data.iloc[:, :-1]
test_y = test_data["label"]

params = {
    "learning_rate": 0.1,
    "boosting_type":"gbdt",
    "max_depth":3,
    "lambda_l2" : 200,
    "metric": "rmse"
}

lgb_train = lgb.Dataset(train_x, train_y)
lgb_test = lgb.Dataset(test_x, test_y)

gbm = lgb.train(params, lgb_train, num_boost_round=300, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=30)