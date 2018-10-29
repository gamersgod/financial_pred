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
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


data_dir = "data/"

data = pd.read_csv(data_dir + "train_data.csv")
data = data.dropna(axis=0)

data_x = data.drop(["fund_name", "label"], axis=1).values
data_y = data["label"].values

print(data_x.shape)

test_data = pd.read_csv(data_dir + "test_data.csv")
test_x = test_data.drop("fund_name", axis=1).values

print(test_x.shape)

N = 5
kf = KFold(n_splits=N, random_state=42, shuffle=True)


def train_sklearn_kflod(model):
    submits = np.zeros([test_x.shape[0], N])
    for k, (train_idx, test_idx) in enumerate(kf.split(data_x, data_y)):
        print("current flod:", k)
        train_x, val_x, train_y, val_y = data_x[train_idx], data_x[test_idx], data_y[train_idx], data_y[test_idx]
        model.fit(train_x, train_y)
        print("train loss", mean_squared_error(model.predict(train_x), train_y))
        print("valid loss", mean_squared_error(model.predict(val_x), val_y))
        submits[:, k] = model.predict(test_x)

    test_data["2018-03-19"] = np.mean(submits, axis=1)
    test_data[["fund_name", "2018-03-19"]].to_csv("sub_reg_5fold_v3.csv", index=None)


def train_sklearn(model):
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)
    print("start train")
    model.fit(train_x, train_y)
    print("train loss", mean_squared_error(model.predict(train_x), train_y))
    print("valid loss", mean_squared_error(model.predict(val_x), val_y))
    test_data["2018-03-19"] = model.predict(test_x)
    test_data[["fund_name", "2018-03-19"]].to_csv("sub_reg_v3.csv", index=None)


def train_dnn():
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2)
    model = Sequential()
    model.add(Dense(600, activation="relu", input_dim=train_x.shape[1]))
    model.add(Dense(300, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    model.compile(loss="mse", metrics=["mae"], optimizer="adam")

    model.summary()
    model.fit(train_x, train_y, batch_size=128, epochs=5, validation_data=(val_x, val_y))


if __name__ == '__main__':
    model = XGBRegressor()
    # model = LGBMRegressor()
    train_sklearn(model)
