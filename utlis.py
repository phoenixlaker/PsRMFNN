import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


def Data_loader(file_x, file_y, size):
    data_x = pd.read_csv(file_x, header=0).values
    data_y = pd.read_csv(file_y, header=0).values
    xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=size)
    sc = StandardScaler().fit(xtrain)
    x_train = sc.transform(xtrain)
    x_test = sc.transform(xtest)
    return x_train, x_test, ytrain, ytest


def Error_cal(y_pred, y_exp):
    test_error = y_pred - y_exp
    E_mse = np.mean(test_error * test_error)
    E_mae = np.mean(np.abs(test_error))
    E_rmse = np.sqrt(E_mse)
    E_mape = np.mean(np.abs(test_error) / y_exp) * 100
    mr = y_pred / y_exp
    E_mean = np.mean(mr)
    h = mr - E_mean
    E_cov = np.sqrt(np.mean(h * h)) / E_mean
    y_exp_mean = np.mean(y_exp)
    y_pred_mean = np.mean(y_pred)
    y_exp_e = y_exp - y_exp_mean
    y_pred_e = y_pred - y_pred_mean
    E_r2 = np.power(np.sum(y_exp_e * y_pred_e), 2) / (np.sum(y_exp_e * y_exp_e) * np.sum(y_pred_e * y_pred_e))
    Err = np.array([E_r2, E_mae, E_rmse, E_mape, E_mean, E_cov])
    Err = np.reshape(Err, (1, 6))
    return Err
