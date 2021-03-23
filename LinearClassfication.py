# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 09:03:43 2021

@author: JerryDai
"""
# In[] import package
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# In[] Build model
def run_gradient_descent(model,
                         batch_size=64,
                         learning_rate=0.01,
                         weight_decay=0,
                         num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iters, losses = [], []
    iters_sub, train_acc, val_acc, test_acc, train_acc_top5, val_acc_top5, test_acc_top5 = [], [], [], [], [], [], []
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True)
    # training
    n = 0 # the number of iterations
    for epoch in range(num_epochs):
        for xs, ts in iter(train_loader):
            if len(ts) != batch_size:
                continue
            zs = model(xs)
            ts = torch.tensor(ts, dtype=torch.long)
            loss = criterion(zs, ts) # compute the total loss
            loss.backward() # compute updates for each parameter
            optimizer.step() # make the updates for each parameter
            optimizer.zero_grad() # a clean up step for PyTorch
            
            # save the current training information
            iters.append(n)
            losses.append(float(loss)/batch_size) # compute *average* loss
            
            if n % 10 == 0:
                print('n:', n)
                iters_sub.append(n)
                train_acc.append(get_accuracy_top1(model, train_data)) # train acc top1
                val_acc.append(get_accuracy_top1(model, val_data)) # val acc top1
                test_acc.append(get_accuracy_top1(model, test_data)) # test acc top1
                # train_acc_top5.append(get_accuracy_top5(model, train_data))  # train acc top5
                # val_acc_top5.append(get_accuracy_top5(model, val_data)) # val acc top5
                # test_acc_top5.append(get_accuracy_top5(model, test_data)) # test acc top5
            # increment the iteration number
            n += 1
    print('The Top-1 accuaray of Linear perceptron on training set', np.mean(train_acc))
    print('The Top-1 accuaray of Linear perceptron Classifier on validation set', np.mean(val_acc))
    print('The Top-1 accuaray of Linear perceptron Classifier on testing set', np.mean(test_acc))
    # print('The Top-5 accuaray of Linear perceptron on training set', np.mean(train_acc_top5))
    # print('The Top-5 accuaray of Linear perceptron Classifier on validation set', np.mean(val_acc_top5))
    # print('The Top-5 accuaray of Linear perceptron Classifier on testing set', np.mean(test_acc_top5))
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

def get_accuracy_top1(model, data):
    loader = torch.utils.data.DataLoader(data, batch_size=500)
    correct, total = 0, 0
    for xs, ts in loader:
        zs = model(xs)
        ts = torch.tensor(ts, dtype=torch.long)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        correct += pred.eq(ts.view_as(pred)).sum().item()
        print('correct:', correct)
        total += int(ts.shape[0])
        return correct / total
    
# def get_accuracy_top5(model, data):
#     loader = torch.utils.data.DataLoader(data, batch_size=500)
#     correct, total = 0, 0
#     for xs, ts in loader:
#         zs = model(xs)
#         ts = torch.tensor(ts, dtype=torch.long)
#         pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
#         pred_top5 = torch.topk(zs, 5, dim = 1)[0]
#         print('pred_top5:', pred_top5)
#         for i in pred_top5:
#             if i in ts.view_as(i):
#                 correct += pred.eq(ts.view_as(pred)).sum().item()
#         print('correct:', correct)
#         total += int(ts.shape[0])
#         return correct / total
    
    
# In[] Load Data
train_path_list = pd.read_csv('train.txt', header = None, sep = ' ', names = ['image_path', 'label'])
val_path_list = pd.read_csv('val.txt', header = None, sep = ' ', names = ['image_path', 'label'])
test_path_list = pd.read_csv('test.txt', header = None, sep = ' ', names = ['image_path', 'label'])

# In[] Exact feature by global color histogram
# train_col_hist =  np.zeros(768)
# for index, tmp_pic_path in enumerate(train_path_list.loc[:, 'image_path']):
#     print(index)
#     temp_pic = cv2.imread(tmp_pic_path)
#     colors = ('b', 'g', 'r')
    
#     tmp_hist_array = np.array([])
#     for i, col in enumerate(colors):
#         hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
#         hist = hist.flatten()
#         tmp_hist_array = np.append(tmp_hist_array, hist)
    
#     train_col_hist = np.vstack((train_col_hist, tmp_hist_array))
    
# train_col_hist = np.delete(train_col_hist, 0, axis = 0)


# val_col_hist =  np.zeros(768)
# for index, tmp_pic_path in enumerate(val_path_list.loc[:, 'image_path']):
#     print(index)
#     temp_pic = cv2.imread(tmp_pic_path)
#     colors = ('b', 'g', 'r')
    
#     tmp_hist_array = np.array([])
#     for i, col in enumerate(colors):
#         hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
#         hist = hist.flatten()
#         tmp_hist_array = np.append(tmp_hist_array, hist)
    
#     val_col_hist = np.vstack((val_col_hist, tmp_hist_array))
    
# val_col_hist = np.delete(val_col_hist, 0, axis = 0)


# test_col_hist =  np.zeros(768)
# for index, tmp_pic_path in enumerate(test_path_list.loc[:, 'image_path']):
#     print(index)
#     temp_pic = cv2.imread(tmp_pic_path)
#     colors = ('b', 'g', 'r')
    
#     tmp_hist_array = np.array([])
#     for i, col in enumerate(colors):
#         hist = cv2.calcHist([temp_pic], [i], None, [256], [0, 256])
#         hist = hist.flatten()
#         tmp_hist_array = np.append(tmp_hist_array, hist)
    
#     test_col_hist = np.vstack((test_col_hist, tmp_hist_array))
    
# test_col_hist = np.delete(test_col_hist, 0, axis = 0)

train_col_hist = np.load('train_col_hist.npy')
val_col_hist = np.load('val_col_hist.npy')
test_col_hist = np.load('test_col_hist.npy')


# In[] Map feature and label
train_col_hist = torch.tensor(train_col_hist, dtype=torch.float)
train_label = torch.tensor(np.array(train_path_list.iloc[:, 1].values), dtype=torch.float)
train_data = TensorDataset(train_col_hist, train_label)

val_col_hist = torch.tensor(val_col_hist, dtype=torch.float)
val_label = torch.tensor(np.array(val_path_list.iloc[:, 1].values), dtype=torch.float)
val_data = TensorDataset(val_col_hist, val_label)

test_col_hist = torch.tensor(test_col_hist, dtype=torch.float)
test_label = torch.tensor(np.array(test_path_list.iloc[:, 1].values), dtype=torch.float)
test_data = TensorDataset(test_col_hist, test_label)
    
# In[] Train model
model = nn.Linear(768, 50)
run_gradient_descent(model, batch_size=256, learning_rate=0.01, num_epochs=10)

# In[] RandomForest
from sklearn.ensemble import RandomForestClassifier

train_X = train_col_hist.numpy()
train_Y = train_label.numpy()

val_X = val_col_hist.numpy()
val_Y = val_label.numpy()

test_X = test_col_hist.numpy()
test_Y = test_label.numpy()

rf_model = RandomForestClassifier(n_estimators = 100)
rf_model.fit(train_X, train_Y)
print('The accuaray of Random Forest Classifier on training set', rf_model.score(train_X, train_Y))
print('The accuaray of Random Forest Classifier on validation set', rf_model.score(val_X, val_Y))
print('The accuaray of Random Forest Classifier on testing set', rf_model.score(test_X, test_Y))

# In[] XGBoost
from xgboost import XGBClassifier
params = {'tree_method':'gpu_hist', 'predictor':'gpu_predictor', 'max_depth': 2, 'n_estimators':20, 'learning_rate':0.1, 'early_stopping_rounds':5, 'nthread':6}
xgbc = XGBClassifier(**params)
     
xgbc.fit(train_X, train_Y)
print('The accuaray of eXtreme Gradient Boosting Classifier on training set', xgbc.score(train_X, train_Y))
print('The accuaray of eXtreme Gradient Boosting Classifier on validation set', xgbc.score(val_X, val_Y))
print('The accuaray of eXtreme Gradient Boosting Classifier on testing set', xgbc.score(test_X, test_Y))


