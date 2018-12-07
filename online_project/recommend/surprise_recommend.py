from surprise import SVDpp, SVD, NMF, SlopeOne, KNNBasic, KNNWithMeans, CoClustering, BaselineOnly
from surprise import Dataset, Reader
from surprise import NormalPredictor, BaselineOnly
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
import pandas as pd
import time
from surprise import accuracy
from surprise.model_selection import GridSearchCV
import os
from collections import defaultdict

file_path = os.path.expanduser('../../_dataset/recommend/ml-100k/u.data')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


def grid():
    raw_ratings = data.raw_ratings
    threshold = int(.9 * len(raw_ratings))
    A_raw_ratings = raw_ratings[:threshold]
    B_raw_ratings = raw_ratings[threshold:]

    data.raw_ratings = A_raw_ratings
    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
    grid_search = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3)
    grid_search.fit(data)
    algo = grid_search.best_estimator['rmse']

    # retrain on the whole set A
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Compute biased accuracy on A
    predictions = algo.test(trainset.build_testset())
    print('Biased accuracy on A,', end='   ')
    accuracy.rmse(predictions)

    # Compute unbiased accuracy on B
    testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
    predictions = algo.test(testset)
    print('Unbiased accuracy on B,', end=' ')
    accuracy.rmse(predictions)


def bench_mark():
    algos = [SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, CoClustering, BaselineOnly]
    for algo in algos:
        begin = time.time()
        cross_validate(algo(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        duration = int((time.time() - begin) * 1000)
        print("time cost = {}ms".format(duration))


bench_mark()
