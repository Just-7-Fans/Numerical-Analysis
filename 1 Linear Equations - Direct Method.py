import numpy
import numpy.matlib

# Hilbert Matrix
n = 30
matrix = numpy.empty([n, n], dtype = float, order = 'C')
for i in range(0, n):
    for j in range(0, n):
        matrix[i][j] = 1 / (i + j + 1)

x = numpy.ones([n, 1], dtype = float, order = 'C') # Assigned Value: [1, ..., 1]
b = numpy.matmul(matrix, x)


# Tikhonov Regularization
C = 0.000000001
b = numpy.matmul(matrix.T, b)
matrix = numpy.matmul(matrix.T, matrix) + C * numpy.matlib.identity(n) # matrix = matrix.T here
matrix = numpy.array(matrix) # change from matrix to ndarray


def Upper_Triangle(U, b):
    #x = numpy.zeros([n, 1], dtype = float, order = 'C')
    for i in range(0, n):  # the ith row
        i = n - 1 - i  # reverse the order: from bottom to top
        sum = 0
        for j in range(i + 1, n):
            sum += U[i][j] * x[j]
        x[i] = (b[i] - sum) / U[i][i]
    return x

def Lower_Triangle(L, b):
    #x = numpy.zeros([n, 1], dtype = float, order = 'C')
    for i in range(0, n):  # the ith row
        sum = 0
        for j in range(0, i):
            sum += L[i][j] * x[j]
        x[i] = (b[i] - sum) / L[i][i]
    return x

def swap(a,b):
    temp = a
    a = b
    b = temp

'''
Problem: x, y = y, x ----- swap(x,y)
Parallel Assignment X '''
def Gauss_Elimination(A, b):
    for i in range(0, n):  # the ith column

        # choose the column pivot
        max, max_row = A[i][i], i
        for u in range(i + 1, n):  # the uth row
            if(abs(A[u][i]) > max):
                max = abs(A[u][i])
                max_row = u
        if(max_row != i):
            for w in range(i, n):
                #A[i][w], A[max_row][w] = A[max_row][w], A[i][w]
                swap(A[i][w], A[max_row][w]) 
            #b[i], b[max_row] = b[max_row], b[i]
            swap(b[i], b[max_row])

        for j in range(i + 1, n):  # the jth row
            l = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= l * A[i][k]  # elementary transformation 
            b[j] -= l * b[i]
    
    x = Upper_Triangle(A, b)
    return x


''' 
Problem: n > 12 ----- Not A Number
n = 13 ----- 10^-17 ~ 2^-53  Probably Reach Precision Limit '''
def Cholesky_Decomposition(A, b):
    L = numpy.zeros([n, n], dtype = float, order = 'C')
    #LT = numpy.zeros([n, n], dtype = float, order = 'C')
    for i in range(0, n):  # the ith column
        sum = 0
        for j in range(0, i):
            sum += L[i][j] ** 2
        L[i][i] = (A[i][i] - sum) ** 0.5
        #LT[i][i] = L[i][i]
        
        for h in range(i + 1, n):  # the hth row
            sum = 0
            for k in range(0, i):
                sum += L[h][k] * L[i][k]
            L[h][i] = (A[h][i] - sum) / L[i][i]
            #LT[i][h] = L[h][i]
    
    y = Lower_Triangle(L, b)
    x = Upper_Triangle(L.T, y)
    return x

print(Cholesky_Decomposition(matrix, b))        
print(Gauss_Elimination(matrix, b))