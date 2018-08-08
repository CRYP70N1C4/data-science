import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func(x, a, b, c):
    return a * x ** b + c;


xdata = np.array([10, 50, 500, 5000, 50000, 500000], dtype=np.double)
ydata = np.array([7000, 11000, 25000, 80000, 250000, 1000000], dtype=np.double)

#
plt.plot(xdata, ydata, 'b*', label='data')

#
popt, pcov = curve_fit(func, xdata, ydata)
xdata = np.arange(0, 500000, 1, np.int32)
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: parquet_demo.py=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
