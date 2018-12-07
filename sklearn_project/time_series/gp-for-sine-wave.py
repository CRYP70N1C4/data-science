import numpy as np
from sklearn.gaussian_process import GaussianProcess
from scipy.interpolate import CubicSpline


def f(x):
    return x * np.sin(x)


X = np.atleast_2d([0., 1., 2., 3., 5., 6., 7., 8., 9.5]).T
Y = f(X).ravel()
print(X)
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

print(x)

gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1,
                     random_start=100)

gp.fit(X,Y)

y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)
print(sigma)