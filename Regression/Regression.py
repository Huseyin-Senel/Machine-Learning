import numpy as np
from matplotlib import pyplot as plt
import random



"""Regression function"""   #Copilot
def regression(x, y):
    """
    :param x: x-axis
    :param y: y-axis
    :return: slope, y-intercept
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = 0
    denominator = 0
    for i in range(len(x)):
        numerator += (x[i] - x_mean) * (y[i] - y_mean)
        denominator += (x[i] - x_mean) ** 2
    slope = numerator / denominator
    y_intercept = y_mean - slope * x_mean
    return slope, y_intercept



x =  np.array(list(range(1, 100)))    #Me :)
y =  np.array(list(range(1, 100)))
for i in range(len(x)):
    y[i]= x[i]+random.randint(0,50)



"""plot regression line"""    #Copilot
slope, y_intercept = regression(x, y)
x_line = np.linspace(0, 200, 200)
y_line = slope * x_line + y_intercept
plt.plot(x_line, y_line, 'r')
#plt.xlim([0, 200])
#plt.ylim([0, 200])
plt.scatter(x, y)
plt.show()

