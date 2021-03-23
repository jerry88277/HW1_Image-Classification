# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:03:43 2021

@author: JerryDai
"""
# In[] import package
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# In[] Load Data
train_path_list = pd.read_csv('train.txt', header = None, sep = ' ', names = ['image_path', 'label'])
val_path_list = pd.read_csv('val.txt', header = None, sep = ' ', names = ['image_path', 'label'])
test_path_list = pd.read_csv('test.txt', header = None, sep = ' ', names = ['image_path', 'label'])


train_col_hist =  np.zeros(768)
for index, tmp_pic_path in enumerate(train_path_list.loc[:, 'image_path']):
    print(index)
    temp_pic = cv2.imread(tmp_pic_path)
    colors = ('b', 'g', 'r')
    
    tmp_hist_array = np.array([])
    for i, col in enumerate(colors):
        hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
        hist = hist.flatten()
        tmp_hist_array = np.append(tmp_hist_array, hist)
    
    train_col_hist = np.vstack((train_col_hist, tmp_hist_array))
    
train_col_hist = np.delete(train_col_hist, 0, axis = 0)


val_col_hist =  np.zeros(768)
for index, tmp_pic_path in enumerate(val_path_list.loc[:, 'image_path']):
    print(index)
    temp_pic = cv2.imread(tmp_pic_path)
    colors = ('b', 'g', 'r')
    
    tmp_hist_array = np.array([])
    for i, col in enumerate(colors):
        hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
        hist = hist.flatten()
        tmp_hist_array = np.append(tmp_hist_array, hist)
    
    val_col_hist = np.vstack((val_col_hist, tmp_hist_array))
    
val_col_hist = np.delete(val_col_hist, 0, axis = 0)


test_col_hist =  np.zeros(768)
for index, tmp_pic_path in enumerate(test_path_list.loc[:, 'image_path']):
    print(index)
    temp_pic = cv2.imread(tmp_pic_path)
    colors = ('b', 'g', 'r')
    
    tmp_hist_array = np.array([])
    for i, col in enumerate(colors):
        hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
        hist = hist.flatten()
        tmp_hist_array = np.append(tmp_hist_array, hist)
    
    test_col_hist = np.vstack((test_col_hist, tmp_hist_array))
    
    # if index == 10000:
    #     break
test_col_hist = np.delete(test_col_hist, 0, axis = 0)



# In[] Test

train_col_hist = torch.tensor(train_col_hist, dtype=torch.float)
train_label = torch.tensor(np.array(train_path_list.iloc[:, 1].values), dtype=torch.float)
train_data = TensorDataset(train_col_hist, train_label)

val_col_hist = torch.tensor(val_col_hist, dtype=torch.float)
val_label = torch.tensor(np.array(val_path_list.iloc[:, 1].values), dtype=torch.float)
val_data = TensorDataset(val_col_hist, val_label)

test_col_hist = torch.tensor(test_col_hist, dtype=torch.float)
test_label = torch.tensor(np.array(test_path_list.iloc[:, 1].values), dtype=torch.float)
test_data = TensorDataset(test_col_hist, test_label)


# In[] train model
def run_gradient_descent(model,
                         batch_size=64,
                         learning_rate=0.01,
                         weight_decay=0,
                         num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iters, losses = [], []
    iters_sub, train_acc, val_acc = [], [] ,[]
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)
    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for xs, ts in iter(train_loader):
            if len(ts) != batch_size:
                continue
            # xs = xs.view(-1, 784) # flatten the image. The -1 is a wildcard
            zs = model(xs)
            ts = torch.tensor(ts, dtype=torch.long)
            loss = criterion(zs, ts) # compute the total loss
            # loss = torch.tensor(loss, dtype=torch.long)
            loss.backward() # compute updates for each parameter
            optimizer.step() # make the updates for each parameter
            optimizer.zero_grad() # a clean up step for PyTorch
            
            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size) # compute *average* loss
            
            if n % 10 == 0:
                print(n)
                iters_sub.append(n)
                train_acc.append(get_accuracy(model, train_data))
                val_acc.append(get_accuracy(model, val_data))
            # increment the iteration number
            n += 1
    # plotting
    plt.figure()
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters, losses, label="Train loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()
    
    plt.figure()
    plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, learning_rate))
    plt.plot(iters_sub, train_acc, label="Train")
    plt.plot(iters_sub, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    return model


def get_accuracy(model, data):
    loader = torch.utils.data.DataLoader(data, batch_size=500)
    correct, total = 0, 0
    for xs, ts in loader:
        # xs = xs.view(-1, 784) # flatten the image
        zs = model(xs)
        ts = torch.tensor(ts, dtype=torch.long)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        return correct / total
        
model = nn.Linear(768, 50)
run_gradient_descent(model, batch_size=32, learning_rate=0.01, num_epochs=3)

