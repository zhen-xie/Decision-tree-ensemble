import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

from sklearn import datasets
from sklearn.model_selection import train_test_split

X_train, y_train = datasets.load_svmlight_file("./input/letter")
X = X_train.todense()
y = y_train

y = y / np.max(y)

X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=1)

dtrain = xgb.DMatrix(X_train, label=y_train)
deval = xgb.DMatrix(X_eval, label=y_eval)

with open("data_letter.txt", "w+") as f:
    f.writelines(str(X.shape[0]))
    f.writelines('\n')
    f.writelines(str(X.shape[1] + 1))
    f.writelines('\n')
    f.writelines('-999.000')
    f.writelines('\n')
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f.writelines(str(X[i,j]))
            f.writelines('\n')
        f.writelines(str(y[i]))
        f.writelines('\n')

depth = 4
trees = 150

param = {'max_depth': depth, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'nthread': 24,
        'eval_metric':['aucpr','auc']}
bst = xgb.train(param, dtrain, num_boost_round=trees,
                evals = [(deval, 'validate'), (dtrain, 'train')], verbose_eval=20)
bst.attributes()

bst.dump_model('final_model01.raw_json.txt', with_stats=True, dump_format='json')

import json
with open('final_model01.raw_json.txt') as myjson:
    data = myjson.read()

obj = json.loads(data)

depth_tree = depth+1
num_node = pow(2,depth_tree) - 1
num_tree = trees

#values = [[[None]*5]*num_node]*num_tree

values = [None] * num_tree
for i in range(num_tree):
    values[i] = [None] * num_node
    for j in range(num_node):
        values[i][j] = [None] * 5


for i in range(num_tree):
    for j in range(num_node):
        root = obj[i]
        depth = math.ceil(math.log(j+2,2)) - 1
        index = j+1
        mystack = []
        for loop in range(depth):
            mystack.append(index%2)
            index = int(index/2)
        for loop in range(len(mystack)):
            if 'children' in root:
                if mystack.pop() == 0:
                    root = root['children'][0]
                else:
                    root = root['children'][1]
            else:
                root = None
                break
        if root is not None:
            if 'children' in root:
                values[i][j][0] = root['split']
                values[i][j][1] = root['split_condition']
                values[i][j][2] = root['missing']%2
                values[i][j][3] = root['cover']
                values[i][j][4] = 0
            else:
                values[i][j][0] = None
                values[i][j][1] = root['leaf']
                values[i][j][2] = None
                values[i][j][3] = root['cover']
                values[i][j][4] = 1

features=[]
for i in range(num_tree):
    for j in range(num_node):
        if values[i][j][0] is not None:
            if values[i][j][0] not in features:
                features.append(values[i][j][0])

with open("model_letter.txt", "w+") as f:
    f.writelines(str(num_tree))
    f.writelines('\n')
    f.writelines(str(depth_tree))
    f.writelines('\n')
    for i in range(num_tree):
        for j in range(num_node):
            for k in range(5):
                if k == 0 and values[i][j][k]:
                    values[i][j][k] = features.index(values[i][j][k])
                f.writelines(str(values[i][j][k]))
                f.writelines('\n')
