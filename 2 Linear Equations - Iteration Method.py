import numpy
import numpy.matlib

# Hilbert Matrix
n = 50
m = 8
matrix = numpy.empty([n, n], dtype = float, order = 'C')
for i in range(0, n):
    for j in range(0, n):
        matrix[i][j] = 1 / (i + j + 1)

x = numpy.ones([n, 1], dtype = float, order = 'C') # Assigned Value: [1, ..., 1]
b = numpy.matmul(matrix, x)

def Conjugate_Gradient(A, b):
    _x = numpy.zeros([n, 1], dtype = float, order = 'C')
    _x[0] = 1  # random initial value
    _r = b - numpy.matmul(A, _x)  # initial residual
    _p = _r  # initial search direction
    k1, k2 = 0, 0

    for i in range(0, 8):
        tempVector = numpy.matmul(A, _p)
        tempProduct = numpy.vdot(_r, _r)

        k1 = tempProduct/numpy.vdot(tempVector, _p)
        x_ = _x + k1 * _p
        r_ = _r - k1 * tempVector
        k2 = numpy.vdot(r_, r_)/tempProduct
        p_ = r_ + k2 * _p
        _x, _r, _p = x_, r_, p_

    return _x

x0 = numpy.zeros([n, 1], dtype = float, order = 'C')
x0[0] = 1
r0 = b - numpy.matmul(matrix, x0)
r0norm = numpy.linalg.norm(r0)
I = numpy.array(numpy.matlib.identity(m + 1))

# Turn any real matrix into Upper Hessenberg Matrix
H = numpy.zeros([n, n], dtype = float, order = 'C')
V = numpy.zeros([n, n], dtype = float, order = 'C')
V[0] = r0.T / r0norm

def Hessenberg(A):
    for i in range(0, m): # the ith column
        Temp = numpy.matmul(A, V[i]) # V[i] stands for the ith row
        for j in range(0, i + 1):
            H[j][i] = numpy.vdot(Temp, V[j])
        for k in range(0, i + 1):
            Temp -= H[k][i] * V[k]
        H[i + 1][i] = numpy.linalg.norm(Temp) # norm
        V[i + 1] = Temp / H[i + 1][i]
    
    for i in range(0, m + 1): 
        H[i][m] = numpy.vdot(numpy.matmul(A, V[m]), V[i])
    return H[0 : m + 1, 0 : m]
    # m + 1

def Least_Square(A, b):
    S = numpy.matmul(A.T, A)
    b = numpy.matmul(A.T, b)
    return numpy.linalg.solve(S, b) # A'Ax = A'b 


''' 
Problem: Why Reshape instead of Transpose is Effective '''
def GMRES(A):
    H = Hessenberg(A)
    y = Least_Square(H, r0norm * I[0])
    W = V.T[0 : n, 0 : m]
    return x0 + (numpy.matmul(W, y)).reshape(n, 1)

print(GMRES(matrix))
print(Conjugate_Gradient(matrix, b))