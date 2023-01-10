import numpy
import matplotlib.pyplot as plt
import math

t, h = 100, 0.1
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
    e = 10 ** -7
    while(True):
        x1, x2 = _x + _X1 * h / 4 + c1 * _X2 * h, _x + _X2 * h / 4 + c2 * _X1 * h
        y1, y2 = _y + _Y1 * h / 4 + c1 * _Y2 * h, _y + _Y2 * h / 4 + c2 * _Y1 * h
        X1_, X2_ = -y1, -y2
        Y1_, Y2_ = x1, x2 
        if(abs(X1_ - _X1) < e and abs(X2_ - _X2) < e  and abs(Y1_ - _Y1) < e and abs(Y2_ - _Y2) < e):
            break
        _X1, _X2 = X1_, X2_
        _Y1, _Y2 = Y1_, Y2_
    return [_x + h * (X1_ + X2_) / 2, _y + h * (Y1_ + Y2_) / 2]

N = int(t / h) + 1
X, X0 = numpy.zeros([N, 1]), numpy.zeros([N, 1])
Y, Y0 = numpy.zeros([N, 1]), numpy.zeros([N, 1])
Z = numpy.zeros([N, 1])
x0 = numpy.linspace(0, 100, N)

for i in range(0, N):
    X0[i], Y0[i] = - math.sin(i * h), math.cos(i * h)

X[0], Y[0], Z[0] = 0, 1, 0.5
for i in range(0, N - 1):
    X[i + 1], Y[i + 1] = ERunge_Kutta(X[i], Y[i])
    # X[i + 1], Y[i + 1] = IRunge_Kutta(X[i], Y[i])
    Z[i + 1] = (X[i + 1] ** 2 + Y[i + 1] ** 2) / 2

# plt.plot(X, Y)
# plt.plot(x0, X, label = r'$p(t)$')
plt.plot(x0, X - X0, label = r'$p(t)_\mathrm{error}$')
plt.plot(x0, Y - Y0, label = r'$q(t)_\mathrm{error}$')
# plt.plot(x0, X0 * 10 ** -9, label = r'$-10^{-9}\sin t$')
# plt.plot(x0, Y0 * 10 ** -9, label = r'$10^{-9}\cos t$')
plt.title(r'$h=$' + str(h))
plt.legend()
plt.show()

plt.plot(x0, 0.5 - Z, label = r'$\frac{p^2+q^2}{2}_\mathrm{error}$')
plt.title(r'$h=$' + str(h))
plt.legend()
plt.show()