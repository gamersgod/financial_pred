import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_data0 = pd.read_csv("data/train_data_0.csv").iloc[:, 1:]
train_data1 = pd.read_csv("data/train_data_1.csv").iloc[:, 1:]
train_data2 = pd.read_csv("data/train_data_2.csv").iloc[:, 1:]
train_data3 = pd.read_csv("data/train_data_3.csv").iloc[:, 1:]
train_data4 = pd.read_csv("data/train_data_4.csv").iloc[:, 1:]
train_data5 = pd.read_csv("data/train_data_5.csv").iloc[:, 1:]
train_data6 = pd.read_csv("data/train_data_6.csv").iloc[:, 1:]
train_data7 = pd.read_csv("data/train_data_7.csv").iloc[:, 1:]
train_data8 = pd.read_csv("data/train_data_8.csv").iloc[:, 1:]
train_data9 = pd.read_csv("data/train_data_9.csv").iloc[:, 1:]
train_data10 = pd.read_csv("data/train_data_10.csv").iloc[:, 1:]

test_data0 = pd.read_csv("data/test_data_0.csv").iloc[:, 1:]
test_data1 = pd.read_csv("data/test_data_1.csv").iloc[:, 1:]
test_data2 = pd.read_csv("data/test_data_2.csv").iloc[:, 1:]

data = pd.concat([train_data0, train_data1, train_data2, train_data3, train_data4, train_data5, train_data6,
                  train_data7, train_data8, train_data9, train_data10, test_data0, test_data1, test_data2],
                 axis=0)

data_x = data.iloc[:, :-1]
data_y = data["label"]

train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)

test_x = pd.read_csv("data/test_data_7.csv").iloc[:, 1:]

print("train size", train_x.shape)
print("val size", val_x.shape)
print("test size", test_x.shape)

def train_lgb():

    params = {
        "learning_rate": 0.5,
        "boosting_type":"gbdt",
        "max_depth":5,
        # "lambda_l2" : 100,
        "lambda_l1" : 10,
        "metric": "rmse"
    }

    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_val = lgb.Dataset(val_x, val_y)

    gbm = lgb.train(params, lgb_train, num_boost_round=2000, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=30)

    test_data = pd.read_csv("data/test_data_7.csv")

    test_x = test_data.iloc[:, 1:]

    test_data["2018-03-19"] = gbm.predict(test_x)

    test_data[["fund_name", "2018-03-19"]].to_csv("submission1.csv", index=None)


def train_xgb():
    params = {
        "eta" : 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0,
        "lambda": 0,
        "eval_metric":"rmse",
        "silent":1,
        "seed": 888
    }

    train_data = xgb.DMatrix(train_x, train_y)
    val_data = xgb.DMatrix(val_x, val_y)
    model = xgb.train(params, train_data, num_boost_round=1000, evals=[(train_data, "train"), (val_data, "val")],
              early_stopping_rounds=50, verbose_eval=50)

    fscore = model.get_fscore()
    fname = []
    score = []
    for k, v in fscore.items():
        fname.append(k)
        score.append(v)
    pd.DataFrame({"fname": fname, "score": score}).to_csv("feat_score.csv", index=None)


def train_cat():
    model = CatBoostRegressor(iterations=2000, depth=5, learning_rate=0.7, loss_function="RMSE")
    model.fit(train_x, train_y, eval_set=(val_x, val_y), plot=True)


if __name__ == '__main__':
    train_cat()
