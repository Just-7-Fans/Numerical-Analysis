import numpy
import matplotlib.pyplot as plt
import math

n, h = 50, 0.01
# Explicit Runge-Kutta (Step:4 / Order: 4)
# y[n + 1] = y[n] + h(k[1] + 2k[2] + 2k[3] + k[4])/6
# k[1] = f(t[n], y[n])      k[2] = f(t[n] + h/2, y[n] + hk[1]/2)
# k[3] = f(t[n] + h/2, y[n] + hk[2]/2)      k[4] = f(t[n] + h, y[n] + hk[3])

def ERunge_Kutta(_x, _y):
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

# Implicit Runge-Kutta (Step:3 / Order: 4)
# y[n + 1] = y[n] + h(k[1]/2 + k[2]/2)
# k[1] = f(t[n] + (1/2 - sqrt(3)/6)h, y[n] + hk[1]/4 + (1/4 - sqrt(3)/6)hk[2])
# k[2] = f(t[n] + (1/2 - sqrt(3)/6)h, y[n] + hk[2]/4 + (1/4 + sqrt(3)/6)hk[1])

def IRunge_Kutta(_x, _y):
    _X1, _X2, _Y1, _Y2 = -_y, -_y, _x, _x
    c1 = 1/4 - math.sqrt(3)/6
    c2 = 1/4 + math.sqrt(3)/6
    while(True):
        x1, x2 = _x + _X1 * h / 4 + c1 * _X2 * h, _x + _X2 * h / 4 + c2 * _X1 * h
        y1, y2 = _y + _Y1 * h / 4 + c1 * _Y2 * h, _y + _Y2 * h / 4 + c2 * _Y1 * h
        X1_, X2_ = -y1, -y2
        Y1_, Y2_ = x1, x2 
        if(abs(X1_ - _X1) < 0.1 and abs(X2_ - _X2) < 0.1 and abs(Y1_ - _Y1) < 0.1 and abs(Y2_ - _Y2) < 0.1):
            break
        _X1, _X2 = X1_, X2_
        _Y1, _Y2 = Y1_, Y2_
    return [_x + h * (X1_ + X2_) / 2, _y + h * (Y1_ + Y2_) / 2]

N = int(n / h)
X, Y = numpy.zeros([N, 1]), numpy.zeros([N, 1])

X[0], Y[0] = 0, 1
for i in range(0, N - 1):
    X[i + 1], Y[i + 1] = ERunge_Kutta(X[i], Y[i])
plt.plot(X, Y)
plt.show()

X[0], Y[0] = 0, 1
for i in range(0, N - 1):
    X[i + 1], Y[i + 1] = IRunge_Kutta(X[i], Y[i])
plt.plot(X, Y)

#plt.plot(X, label = r'$x_1(t)$')
#plt.plot(Y, label = 'y(t)', color = '#2ca02c')
#plt.legend()
plt.show()