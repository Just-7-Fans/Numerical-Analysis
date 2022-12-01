import numpy
import math
import matplotlib.pyplot as plt

PIE = math.pi
def ln(x):
    return math.log(x)
# f(x) = x^2 * ln(x+2)
def f(x):
    return x ** 2 * ln(x + 2)
l0 = 3 * ln(3) - 26 / 9
l1 = -15 * ln(3) / 4 + 13 / 3
l2 = 42 * ln(3) / 5 - 2042 / 225
l3 = -165 * ln(3) / 8 + 409 / 18
c0, c1, c2, c3 = 0.923490, 0.636377, 0.432539, 0.220737

La0, La1, La2, La3 = l0 / 2, 3 * l1 / 2, 5 * l2 / 2, 7 * l3 / 2
Lq0, Lq1, Lq2, Lq3 = La0 - La2 / 2, La1 - La3 * 3 / 2, 3 * La2 / 2, 5 * La3 / 2
Ta0, Ta1, Ta2, Ta3 = c0 / PIE, 2 * c1 / PIE, 2 * c2 / PIE, 2 * c3 / PIE
Tq0, Tq1, Tq2, Tq3 = Ta0 - Ta2, Ta1 - 3 * Ta3, Ta2 * 2, Ta3 * 4
def LegendreSquareA(x):
    return Lq0 + Lq1 * x + Lq2 * x ** 2 + Lq3 * x ** 3
def ChebyshevCloseA(x):
    return Tq0 + Tq1 * x + Tq2 * x ** 2 + Tq3 * x ** 3


# Lagrange Interpolation with Chebyshev Zero Point
n = 3
x0 = numpy.zeros([n + 1, 1])
for i in range(0, n + 1):
    x0[i] = math.cos((2 * i + 1) * PIE / (2 * (n + 1)))
# Lagrange Polynomial
def Lk(x, k):
    s = 1
    for i in range(0, n + 1):
        if(i != k):
            s *= x - x0[i]
            s /= x0[k] - x0[i]
    return s
def L(x):
    sum = 0
    for i in range(0, n + 1):
        sum += Lk(x, i) * f(x0[i])
    return sum


X = numpy.linspace(-1, 1, 201)
F = numpy.zeros([X.size, 1])
for i in range(0, X.size):
    F[i] = f(X[i])
x0 = numpy.linspace(-1, 1, 5)
plt.xticks(x0)
plt.plot(X, F, label = 'Original')
plt.plot(X, LegendreSquareA(X), label = 'Legendre')
#plt.plot(X, ChebyshevCloseA(X), label = 'Chebyshev')
#plt.plot(X, L(X), label = 'Lagrange')
plt.legend()
plt.show()