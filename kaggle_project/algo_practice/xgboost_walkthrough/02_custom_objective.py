import xgboost as xgb
import numpy as np

dtrain = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.train')
dtest = xgb.DMatrix('../../../_dataset/xgboost/agaricus.txt.test')

param = {'max_depth': 2, 'eta': 1, 'silent': 1}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 2


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)


# user define objective function, given prediction, return gradient and second order gradient
# this is log likelihood loss
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

bst = xgb.train(param, dtrain, num_round, watchlist, logregobj, evalerror)
