import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n, h = 50, 0.01
# Runge-Kutta
# y[n + 1] = y[n] + h(k[1] + 2k[2] + 2k[3] + k[4])/6
# k[1] = f(t[n], y[n])      k[2] = f(t[n] + h/2, y[n] + hk[1]/2)
# k[3] = f(t[n] + h/2, y[n] + hk[2]/2)      k[4] = f(t[n] + h, y[n] + hk[3])

def Runge_Kutta(_x, _y):
    x, y = _x, _y
    X1, Y1 = -y, x
    x, y = _x + X1 * h / 2, _y + Y1 * h / 2
    X2, Y2 = -y, x
    x, y = _x + X2 * h / 2, _y + Y2 * h / 2
    X3, Y3 = -y, x
    x, y = _x + X3 * h, _y + Y3 * h
    X4, Y4 = -y, x

    return [_x + h * (X1 + 2 * X2 + 2 * X3 + X4) / 6, \
        _y + h * (Y1 + 2 * Y2 + 2 * Y3 + Y4) / 6]
    
N = int(n / h)
X, Y = numpy.zeros([N, 1]), numpy.zeros([N, 1])
X[0], Y[0] = 0, 1

for i in range(0, N - 1):
    X[i + 1], Y[i + 1] = Runge_Kutta(X[i], Y[i])

plt.plot(X, Y)
plt.plot(X, label = r'$x_1(t)$')
#plt.plot(Y, label = 'y(t)', color = '#2ca02c')
#plt.legend()
plt.show()