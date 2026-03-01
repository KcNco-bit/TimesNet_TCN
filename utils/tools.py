import os

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import json
from scipy import stats
from scipy.stats import norm

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

        print(">>>>-----------------------------------Separator line-----------------------------------<<<<")



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, pred, error):
    """
    Results visualization
    """
    with open('./loss/loss_history.json', 'r') as f:
        loss_data = json.load(f)

    tr_l = loss_data['train_loss']
    v_l = loss_data['val_loss']
    te_l = loss_data['test_loss']
    epochs = range(1, len(tr_l) + 1)

    fig = plt.figure(figsize=(10, 8))

    ax1 = plt.subplot(2, 2, 1) 
    ax1.plot(epochs, tr_l, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, v_l, 'orange', label='Val Loss', linewidth=2) 
    ax1.plot(epochs, te_l, 'r-', label='Test Loss', linewidth=2) 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax2 = plt.subplot(2, 2, 2) 

    preds_flat = pred.flatten()
    trues_flat = true.flatten()
    ax2.scatter(trues_flat, preds_flat, alpha=0.5, s=10)
    
    min_val = min(trues_flat.min(), preds_flat.min())
    max_val = max(trues_flat.max(), preds_flat.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth = 1.5)
    
    ax2.set_xlabel('True values')
    ax2.set_ylabel('Pred values')
    ax2.set_title('Parity plot')


    ax3 = plt.subplot(2, 1, 2)

    preds_flat = pred.flatten()
    trues_flat = true.flatten()

    ax3.plot(trues_flat, 'b-', label='True', linewidth=1.5)
    ax3.plot(preds_flat, 'r--', label='Pred', linewidth=1.5)
    ax3.set_xlabel('Samples')
    ax3.set_ylabel('Value')
    ax3.set_title('Prediction Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show() 


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def Diebold_Mariano_test(error, model, args, h=1):
    """
    Diebold-Mariano Test
    
    Parameters:
        error: Loss of compared model
        model: Name of your own model
        h: Prediction steps
    
    Return:
        dm_stat: Diebold-Mariano statistics
        p_value: p value
    """
    ffp = f'./results/{model}.csv'
    df = pd.read_csv(ffp)

    loss_a = (df['errors'].values) ** 2
    loss_b = error ** 2
    d = loss_a - loss_b

    n = len(d)
    d_mean = np.mean(d)
    
    d_centered = d - d_mean
    
    gamma = np.zeros(h)
    for k in range(h):
        if k == 0:
            gamma[k] = np.mean(d_centered * d_centered)
        else:
            gamma[k] = np.mean(d_centered[k:] * d_centered[:-k])
    
    var_d = gamma[0]
    if h > 1:
        weights = 1 - np.arange(1, h) / h
        var_d += 2 * np.sum(weights * gamma[1:])
    
    if var_d <= 0:
        var_d = np.var(d, ddof=0)
    
    dm_stat = d_mean / np.sqrt(var_d / n)
    p_value = norm.cdf(dm_stat)
    
    if dm_stat < 0:
        if p_value < 0.05:
            print(">>>>-----------------------------------Diebold Mariano Test-----------------------------------<<<<")
            print(f"Diebold-Mariano statistic: {dm_stat:.6f}, P value: {p_value:.6f}.")
            print(f"The {model} model is better than {args.model}.")
        else:
            print(">>>>-----------------------------------Diebold Mariano Test-----------------------------------<<<<")
            print(f"Diebold-Mariano statistic: {dm_stat:.6f}, P value: {p_value:.6f}.")
            print(f"The difference between the two models is not statistically significant.")
    else:
        print(">>>>-----------------------------------Diebold Mariano Test-----------------------------------<<<<")
        print(f"Diebold-Mariano statistic: {dm_stat:.6f}, P value: {p_value:.6f}.")
        print(f"The {model} model is worse than {args.model}.")

    return

class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile    
        if self.quantile < 0 or self.quantile > 1:
            raise ValueError("Quantile must be in range [0, 1]")

    
    def forward(self, input, target):
        error = target - input
        loss = torch.where(error >= 0, self.quantile * error, (self.quantile - 1) * error)

        return torch.mean(loss)
    

def visual_quantile(fith_path):
    
    df = pd.read_csv(fith_path)
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(df))
    plt.fill_between(x, df['lower'], df['upper'], 
                     alpha=0.1, color='blue', label='95% Prediction Interval')
    
    plt.plot(x, df['trues'], 'b-', linewidth=1.5, label='True Values')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Quantile Regression Results')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def calculate_quantile(fith_path, q):
    f = pd.read_csv(fith_path)
    trues = f['trues'].values
    down = f['lower'].values
    up = f['upper'].values

    within_interval = np.logical_and(trues >= down, trues <= up)
    picp = np.mean(within_interval) * 100

    ace = picp - q*100

    interval_widths = up - down
    mpiw = np.mean(interval_widths)

    r = np.max(trues)-np.min(trues)
    pinaw = mpiw/r

    print(f"PICP: {picp:.3f}, ACE: {ace:.3f}, MPIW: {mpiw:.3f}, PINAW: {pinaw:.3f}")
    return
    