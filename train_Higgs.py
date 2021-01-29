import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb

higgs_df = pd.read_csv('input/training_higgs.csv')
higgs_df.head()
higgs_df.info()
'''
pd.set_option('precision',2)
plt.figure(figsize=(10, 8))
sns.heatmap(higgs_df.drop(['Label','EventId'],axis=1).corr(), square=True)
plt.suptitle("Pearson Correlation Heatmap")
#plt.show()
'''

higgs_df_binary_label = pd.get_dummies(higgs_df['Label'])
higgs_df['Label'] = higgs_df_binary_label
corr_with_label = higgs_df.corr()["Label"].sort_values(ascending=False)
'''
plt.figure(figsize=(14,6))
corr_with_label.drop("Label").plot.bar()
#plt.show()
'''

higgs_df.info()

higgs_df = higgs_df.drop(['EventId',], axis=1)
X_higgs = higgs_df.drop(['Label','Weight'],axis=1)
y_higgs = higgs_df['Label']
w_higgs = higgs_df['Weight']

X_higgs.shape[0]
X_higgs.shape[1]
y_higgs.shape


with open("data_Higgs.txt", "w+") as f:
    f.writelines(str(X_higgs.shape[0]))
    f.writelines('\n')
    f.writelines(str(X_higgs.shape[1]))
    f.writelines('\n')
    f.writelines('-999.000')
    f.writelines('\n')
    for i in range(X_higgs.shape[0]):
        for j in range(X_higgs.shape[1]):
            f.writelines(str(X_higgs.iloc[i,j]))
            f.writelines('\n')


col_names = list(X_higgs.columns)
col_names

w_higgs.shape

y_higgs.value_counts()

xg_dmat = xgb.DMatrix(X_higgs, label=y_higgs, missing=-999.000, weight=w_higgs,
                     silent=False, feature_names=col_names, feature_types=None, nthread=-1)

xg_dmat.feature_names
xg_dmat.get_base_margin()
xg_dmat.get_label()
xg_dmat.get_weight()
xg_dmat.num_col()
xg_dmat.num_row()
xg_dmat_training = xg_dmat.slice(X_higgs.index[0:200000])
xg_dmat_validating = xg_dmat.slice(X_higgs.index[200000:2500000])
xg_dmat_training.num_row()
xg_dmat_validating.num_row()

depth = 8
trees = 3000

param = {'max_depth': depth, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic','nthread': 24,
        'eval_metric':['aucpr','auc']}
bst = xgb.train(param, xg_dmat_training, num_boost_round=trees,
                evals = [(xg_dmat_validating, 'validate'), (xg_dmat_training, 'train')], verbose_eval=20)

bst.dump_model('final_model01.raw_json.txt', with_stats=True, dump_format='json')

predictions01 =bst.predict(xg_dmat)


#xgb.plot_tree(bst, num_trees=1)
#plt.show()

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


with open("model_Higgs.txt", "w+") as f:
    f.writelines(str(num_tree))
    f.writelines('\n')
    f.writelines(str(depth_tree))
    f.writelines('\n')
    for i in range(num_tree):
        for j in range(num_node):
            for k in range(5):
                if k == 0 and values[i][j][k]:
                    values[i][j][k] = col_names.index(values[i][j][k])
                f.writelines(str(values[i][j][k]))
                f.writelines('\n')
