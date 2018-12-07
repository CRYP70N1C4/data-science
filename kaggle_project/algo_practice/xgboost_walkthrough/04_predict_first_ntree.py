import xgboost as xgb
import numpy as np

dtrain = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.train')
dtest = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.test')

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round, watchlist)

### predict using first 1 tree
label = dtest.get_label()
ypred1 = bst.predict(dtest, ntree_limit=1)
# by default, we predict using all the trees
ypred2 = bst.predict(dtest)
print('error of ypred1=%f' % (np.sum((ypred1 > 0.5) != label) / float(len(label))))
print('error of ypred2=%f' % (np.sum((ypred2 > 0.5) != label) / float(len(label))))
