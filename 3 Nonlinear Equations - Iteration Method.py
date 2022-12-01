# x^3 + 2x^2 + 10x - 20 = 0
def CubeRoot(x):
    if(x >= 0):
        return x ** (1 / 3)
    else:
        return -(-x) ** (1 / 3)

# Method 1(Diverge)
x = -3.16
for i in range(0, 100):
    x = (20 - 2 * x ** 2 - x ** 3) / 10
print(x)

# Method 2(Diverge)
x = 200
for i in range(0, 100):
    x = CubeRoot(20 - 2 * x ** 2 - 10 * x)
print(x)

# Method 1 -> Steffensen(Converge)
x = 1
for i in range(0, 5):
    y = (20 - 2 * x ** 2 - x ** 3) / 10
    z = (20 - 2 * y ** 2 - y ** 3) / 10
    x = x - (y - x) ** 2 / (z - 2 * y + x)
print(x)

# Method 2 -> Steffensen(Converge)
x = 1
for i in range(0, 8):
    y = CubeRoot(20 - 2 * x ** 2 - 10 * x)
    z = CubeRoot(20 - 2 * y ** 2 - 10 * y)
    x = x - (y - x) ** 2 / (z - 2 * y + x)
print(x)

print("Newton \n")
# Newton(Converge)
x = 1
for i in range(0, 10):
    x = x - (x ** 3 + 2 * x ** 2 + 10 * x - 20)/(3 * x ** 2 + 4 * x + 10)
    print(x)

