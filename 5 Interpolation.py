import numpy
import numpy.matlib
import math
import matplotlib.pyplot as plt

n = 10
x0 = numpy.linspace(-1, 1, n + 1)

def f(x):
    return 1 / (1 + 25 * x ** 2)

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

# Piecewise Linear
def I(x):
    number = (x * n + n) // 2
    number[x.size - 1] = n - 1
    x1, x2 = (number * 2 - n) / n, (number * 2 + 2 - n) / n
    r = (x - x1) / (x2 - x1)
    return (1 - r) * f(x1) + r * f(x2)


# Cubic Spline
def Thomas(A, b):
    L = numpy.array(numpy.matlib.identity(n - 1))
    U = numpy.zeros([n - 1, n - 1], dtype = float, order = 'C')
    U[0][0] = A[0][0]
    for i in range(1, n - 1):
        U[i - 1][i] = A[i - 1][i]
        L[i][i - 1] = A[i][i - 1] / U[i - 1][i - 1]
        U[i][i] = A[i][i] - L[i][i - 1] * U[i - 1][i]
    return Upper_Triangle(U, Lower_Triangle(L, b))
    
def Upper_Triangle(U, b):
    x = numpy.zeros([n - 1, 1], dtype = float, order = 'C')
    for i in range(0, n - 1):  # the ith row
        i = n - 2 - i  # reverse the order: from bottom to top
        sum = 0
        for j in range(i + 1, n - 1):
            sum += U[i][j] * x[j]
        x[i] = (b[i] - sum) / U[i][i]
    return x

def Lower_Triangle(L, b):
    x = numpy.zeros([n - 1, 1], dtype = float, order = 'C')
    for i in range(0, n - 1):  # the ith row
        sum = 0
        for j in range(0, i):
            sum += L[i][j] * x[j]
        x[i] = (b[i] - sum) / L[i][i]
    return x

# 1. S" First
A = numpy.array(numpy.matlib.identity(n - 1))
A = 2 * A
b = numpy.zeros([n - 1, 1], dtype = float, order = 'C')
for i in range(1, n - 1):
    A[i][i - 1], A[i - 1][i] = 0.5, 0.5

k = 2 * (2 / n) ** 2
for i in range(0, n - 1):
    b[i] = 6 * (f(-1 + 2 * (i + 2) / n) - 2 * f(-1 + 2 * (i + 1) / n) + f(-1 + 2 * i / n)) / k
M = numpy.concatenate((numpy.concatenate(([[0]],Thomas(A, b))), [[0]]))

def S(x):
    number = int((x * n + n) // 2)
    #number = number.astype(int)
    if(x == 1):
        number = n - 1
    x1, x2 = (number * 2 - n) / n, (number * 2 + 2 - n) / n
    h = 2 / n
    return M[number] * (x2 - x) ** 3 / (6 * h) + M[number + 1] * (x - x1) ** 3 / (6 * h) + \
        (f(x1) - M[number] * h ** 2 / 6) * (x2 - x) / h + (f(x2) - M[number + 1] * h ** 2 / 6) * (x - x1) / h
# 2. S' First

X = numpy.linspace(-1, 1, 201)
F = f(X)
f0 = f(x0)

plt.subplot(2, 2, 1)
plt.plot(X, F, label = 'f(x)')


plt.subplot(2, 2, 2)
L = L(X)
ymin, ymax = L.min(), L.max()
dy = (ymax - ymin) * 0.1
plt.plot(X, L, label = 'L(x)')
plt.plot(X, F, label = 'f(x)')
plt.xticks(x0)
plt.ylim(ymin - dy, ymax + dy)

# Dots and Lines for Annotation
plt.scatter([x0,], [f0,], 5, color = 'black')
for i in range(0, n + 1):
    plt.plot([x0[i],x0[i]], [ymin - dy, f0[i]], color = 'black', linewidth = 1, linestyle = "--")
plt.legend(loc = 'upper right')


plt.subplot(2, 2, 3)
I = I(X)
ymin, ymax = I.min(), I.max()
dy = (ymax - ymin) * 0.1
plt.plot(X, I, label = 'I(x)')
plt.plot(X, F, label = 'f(x)')
plt.xticks(x0)
plt.ylim(ymin - dy, ymax + dy)

plt.scatter([x0,], [f0,], 5, color = 'black')
for i in range(0, n + 1):
    plt.plot([x0[i],x0[i]], [ymin - dy, f0[i]], color = 'black', linewidth = 1, linestyle = "--")
plt.legend(loc = 'upper right')


plt.subplot(2, 2, 4)
SA = numpy.zeros([X.size, 1], dtype = float, order = 'C')
for i in range(0, X.size):
    SA[i] = S(X[i])

ymin, ymax = SA.min(), SA.max()
dy = (ymax - ymin) * 0.1
plt.plot(X, SA, label = 'S(x)')
plt.plot(X, F, label = 'f(x)')
plt.xticks(x0)
plt.ylim(ymin - dy, ymax + dy)

plt.scatter([x0,], [f0,], 5, color = 'black')
for i in range(0, n + 1):
    plt.plot([x0[i],x0[i]], [ymin - dy, f0[i]], color = 'black', linewidth = 1, linestyle = "--")
plt.legend(loc = 'upper right')
plt.show()