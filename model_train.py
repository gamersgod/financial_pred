import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


#TODO: eluer feature not so good, may be use 1 to - or / it.
data = pd.read_csv("new_data/train_data1_481_30.csv")
data = data.dropna(axis=0)

data_x = data.drop("label", axis=1).values
data_y = data["label"].values

print(data_x.shape)

test_data = pd.read_csv("new_data/test_data.csv")
test_x = test_data.iloc[:, 1:].values

print(test_x.shape)

N = 5
kf = KFold(n_splits=N, random_state=42, shuffle=True)


def train_lgb():

    params = {
        "learning_rate": 0.6,
        "boosting_type":"gbdt",
        "num_leaves":120,
        # "lambda_l2" : 100,
        "lambda_l1" : 10,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "metric": "rmse",
        "verbose": -1
    }

    submits = np.zeros([test_x.shape[0], N])
    for k, (train_idx, test_idx) in enumerate(kf.split(data_x, data_y)):
        train_x, val_x, train_y, val_y = data_x[train_idx], data_x[test_idx], data_y[train_idx], data_y[test_idx]
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_eval = lgb.Dataset(val_x, val_y, reference=lgb_train)

        gbm = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=[lgb_train, lgb_eval],
                        early_stopping_rounds=50, verbose_eval=50)
        # submits[:, k] = gbm.predict(test_x, num_iteration=gbm.best_iteration)

    # test_data["label"] = np.mean(submits, axis=1)
    # test_data[["fund_name", "label"]].to_csv("sub_lgb_kflod5.csv", index=False)


def train_xgb():
    params = {
        "eta" : 0.3,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "alpha": 0,
        "lambda": 0,
        "eval_metric":"rmse",
        "silent":1,
        "seed": 888
    }

    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)
    train_data = xgb.DMatrix(train_x, train_y)
    val_data = xgb.DMatrix(val_x, val_y)
    model = xgb.train(params, train_data, num_boost_round=5000, evals=[(train_data, "train"), (val_data, "val")],
              early_stopping_rounds=50, verbose_eval=50)

    fscore = model.get_fscore()
    fname = []
    score = []
    for k, v in fscore.items():
        fname.append(k)
        score.append(v)
    pd.DataFrame({"fname": fname, "score": score}).to_csv("feat_score.csv", index=None)


def train_cat():
    # train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.3)
    submits = np.zeros([test_x.shape[0], N])
    for k, (train_idx, test_idx) in enumerate(kf.split(data_x, data_y)):
        print("current flod:", k)
        train_x, val_x, train_y, val_y = data_x[train_idx], data_x[test_idx], data_y[train_idx], data_y[test_idx]
        model = CatBoostRegressor(iterations=30000, depth=7, learning_rate=2.0, loss_function="RMSE", task_type="GPU",
                                  use_best_model=True)
        model.fit(train_x, train_y, eval_set=(val_x, val_y))
        submits[:, k] = model.predict(test_x)
    # print("loss", mean_squared_error(model.predict(val_x), val_y))
    test_data["label"] = np.mean(submits, axis=1)
    test_data[["fund_name", "label"]].to_csv("sub_catboost_kfold5_v2.csv", index=False)


def train_sklearn_lgb():
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)
    print("start train")
    model = LGBMRegressor(silent=-1)
    model.fit(train_x, train_y)
    print("train loss", mean_squared_error(model.predict(train_x), train_y))
    print("valid loss", mean_squared_error(model.predict(val_x), val_y))
    test_data["2018-03-19"] = model.predict(test_x)
    test_data[["fund_name", "2018-03-19"]].to_csv("sub_lgb_reg_v1.csv", index=None)


def train_sklearn_lgb_kflod():
    submits = np.zeros([test_x.shape[0], N])
    for k, (train_idx, test_idx) in enumerate(kf.split(data_x, data_y)):
        print("current flod:", k)
        train_x, val_x, train_y, val_y = data_x[train_idx], data_x[test_idx], data_y[train_idx], data_y[test_idx]
        model = LGBMRegressor()
        model.fit(train_x, train_y)
        print("train loss", mean_squared_error(model.predict(train_x), train_y))
        print("valid loss", mean_squared_error(model.predict(val_x), val_y))
        submits[:, k] = model.predict(test_x)

    test_data["2018-03-19"] = np.mean(submits, axis=1)
    test_data[["fund_name", "2018-03-19"]].to_csv("sub_lgb_reg_5fold_v1.csv", index=None)


def train_sklearn():
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)
    print("start train")
    model = RandomForestRegressor(verbose=2)
    model.fit(train_x, train_y)
    print("train loss", mean_squared_error(model.predict(train_x), train_y))
    print("valid loss", mean_squared_error(model.predict(val_x), val_y))
    # test_data["2018-03-19"] = model.predict(test_x)
    # test_data[["fund_name", "2018-03-19"]].to_csv("sub_lgb_reg_v1.csv", index=None)


if __name__ == '__main__':
    train_sklearn_lgb()
