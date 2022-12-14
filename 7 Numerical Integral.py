import numpy
import math

# Gauss-Legendre for [a, b]
# n = 4
x = numpy.zeros([5, 1])
y = numpy.zeros([5, 1])
A = numpy.zeros([5, 1])
x[0], x[1], x[2], x[3], x[4] = -0.9061798459, -0.5384693101, 0, 0.5384693101, 0.9061798459
A[0], A[1], A[2], A[3], A[4] = 0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851

def f(x):
    return math.sin(2 * math.pi / x) / (x ** 2)
def Gauss_Legendre(a, b):
    sum = 0
    for i in range(0, 5):
        # change integral interval: x = (b - a)y/2 + (b + a)/2 -> x in [-1, 1]
        y[i] = x[i] * (b - a) / 2 + (b + a) / 2
        sum += A[i] * f(y[i]) * (b - a) / 2
    return sum

print(Gauss_Legendre(1, 3))
integral = 0
for i in range(0, 4):
    h = 0.5
    integral += Gauss_Legendre(1 + i * h, 1 + (i + 1) * h)
print(integral)

# Romberg 
def trapezoid(k, a, b):
    area = 0
    h = (b - a) / k
    for i in range(0, k):
        area += f(a + i * h) + f(a + (i + 1) * h)
    area *= h / 2
    return area
T = numpy.zeros([10, 10])

T[0][1] = trapezoid(1, 1, 3) 
T[1][1] = trapezoid(2, 1, 3)
T[0][2] = (4 * T[1][1] - T[0][1]) / (4 - 1)
piece = 2
while(abs(T[0][piece] - T[0][piece - 1]) > 10e-7):
    T[piece][1] = trapezoid(2 ** piece, 1, 3)
    i = 1
    while(piece - i >= 0):
        T[piece - i][1 + i] = ((4 ** i) * T[piece - i + 1][i] - T[piece - i][i]) / (4 ** i - 1)
        i += 1
    piece += 1
print(piece)
print(T[0][piece])