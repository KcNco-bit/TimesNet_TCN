import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))

def R2(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def PCC(pred, true):
    pred_centered = pred - np.mean(pred)
    true_centered = true - np.mean(true)
    numerator = np.sum(pred_centered * true_centered)
    denominator = np.sqrt(np.sum(true_centered**2)) * np.sqrt(np.sum(pred_centered**2))
    if denominator == 0:
        return 0.0
    pcc = numerator/denominator
    return pcc

def CCC(pred, true):
    true_mean = np.mean(true)
    pred_mean = np.mean(pred)
    covariance = np.sum((true - true_mean) * (pred - pred_mean))
    var_true = np.sum((true - true_mean)**2)
    var_pred = np.sum((pred - pred_mean)**2)
    mean_diff = len(true) * (true_mean - pred_mean)**2
    numerator = 2 * covariance
    denominator = var_true + var_pred + mean_diff
    if denominator == 0:
        return 0.0
    ccc = numerator/denominator
    return ccc

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    pcc = PCC(pred, true)
    ccc = CCC(pred, true)

    return mae, mse, rmse, mape, mspe, r2, pcc, ccc
