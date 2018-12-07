import xgboost as xgb

dtrain = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.train')
dtest = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.test')
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
xgb.train(param, dtrain, 2, watchlist)

bst = xgb.train(param, dtrain, 1, watchlist)


# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=True, will always give you margin values before logistic transformation
ptrain = bst.predict(dtrain, output_margin=True)
ptest = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)


print('this is result of running from initial prediction')
bst = xgb.train(param, dtrain, 1, watchlist)
