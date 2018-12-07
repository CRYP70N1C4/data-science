import xgboost as xgb
import numpy as np

dtrain = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.train')
dtest = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.test')

# change booster to gblinear, so that we are fitting a linear model
# alpha is the L1 regularizer
# lambda is the L2 regularizer
# you can also set lambda_bias which is L2 regularizer on the bias term
param = {'silent': 1, 'objective': 'binary:logistic', 'booster': 'gblinear',
         'alpha': 0.0001, 'lambda': 1}

watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 4
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dtest)
labels = dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))
