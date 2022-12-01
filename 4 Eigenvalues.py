import numpy
import numpy.matlib

n = 8
matrix = numpy.zeros([n, n], dtype = float, order = 'C')
for i in range(0, n):
    matrix[i][i] = 2
    if(i != 0):
        matrix[i][i - 1] = -1
    if(i != n - 1):
        matrix[i][i + 1] = -1


'''
Classical Jacobi Method
Givens Transform: Eliminate the element of the most modulus (off the diagonal) '''
A1 = matrix
def Givens(A, p, q):
    # Assume p < q
    J = numpy.array(numpy.matlib.identity(n))
    if(A[p][q] == 0):
        c, s = 1, 0
    elif(A[p][p] - A[q][q] == 0):
        c, s = 2 ** 0.5 / 2, 2 ** 0.5 / 2
    else:   
        cot = (A[p][p] - A[q][q]) / (2 * A[p][q]) # cot 2\theta
        t = numpy.sign(cot)/(abs(cot) + (1 + cot ** 2) ** 0.5)
        c = (1 + t ** 2) ** -0.5
        s = c * t
    J[p][p], J[q][q] = c, c
    J[p][q], J[q][p] = s, -s
    return J

def offDiagonal(A):
    max_row, max_column = 0, 0
    max_element = 0
    for i in range(0, n):
        for j in range(0, n):
            if(i != j and abs(A[i][j]) > max_element):
                max_row, max_column = i, j
                max_element = abs(A[i][j])
    return Givens(A, max_row, max_column)

for i in range(0, 100):
    J = offDiagonal(A1)
    A1 = numpy.matmul(numpy.matmul(J, A1), J.T)
#print("Classical Jacobi Method\n", A1)
#print("\n")
print("Approximate Eigenvalues:")
for i in range(0, n):
    print(A1[i][i], " ")
print("\n")


'''
Cycle Jacobi Method
Scan all the elements with a specific threshold w
Parallel Computing  '''
A2 = matrix
w = (2 * n - 2) ** 0.5 / 2
for i in range(0, 10):
    w = w / (i + 1)
    for j in range(0, n):
        for k in range(0, n):
            if(j != k and abs(A2[j][k]) > w):
                J = Givens(A2, j, k)
                A2 = numpy.matmul(numpy.matmul(J, A2), J.T)
#print("Cycle Jacobi Method\n", A2)
#print("\n")
print("Approximate Eigenvalues:")
for i in range(0, n):
    print(A2[i][i], " ")
print("\n")


'''
QR Method
A(k) = Q(k)R(k)     A(k + 1) = R(k)Q(k) = Q(k + 1)R(k + 1)'''
B = matrix
def HessenGivens(A, i):
    J = numpy.array(numpy.matlib.identity(n))
    if(A[i + 1][i] == 0):
        c, s = 1, 0
    elif(abs(A[i + 1][i]) >= abs(A[i][i])):   
        t = A[i][i] / A[i + 1][i]
        s = numpy.sign(A[i + 1][i]) * (1 + t ** 2) ** (-0.5)
        c = s * t
    else:
        t = A[i + 1][i] / A[i][i]
        c = numpy.sign(A[i][i]) * (1 + t ** 2) ** (-0.5)
        s = c * t

    J[i][i], J[i + 1][i + 1] = c, c
    J[i][i + 1], J[i + 1][i] = s, -s
    return J

def HessenQR(A):
    M = numpy.array(numpy.matlib.identity(n))
    for i in range(0, n - 1):
        J = HessenGivens(A, i)
        A = numpy.matmul(J, A)
        M = numpy.matmul(J, M) # Q.T
    return numpy.matmul(A, M.T)

for i in range(0, 100):
    B = HessenQR(B)
#print("QR Method\n", B)
#print("\n")
print("Approximate Eigenvalues:")
for i in range(0, n):
    print(B[i][i], " ")