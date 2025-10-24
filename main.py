import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff

def f(x):
    return (3 * (x**2)) - (63 * x) + 305

def metoda_bisekcji(a, b, m):
    x1 = a
    x2 = b
    x_kroki = np.zeros(m, dtype=float)
    y_kroki = np.zeros(m, dtype=float)
    for i in range(0, m):
        x = (x1 + x2) / 2

        x_kroki[i] = x
        # y_kroki[i] = f(x_kroki[i])

        y = f(x)
        y1 = f(x1)
        if y * y1 > 0:
            x1 = x
        else:
            x2 = x
    return x, x_kroki, y_kroki


def metoda_cieciw(a, b, n):
    x = np.zeros(n, dtype=float)
    cieciwy = np.zeros((n, 2, 2), dtype=float)

    s_x = symbols('s_x')
    eq = (3 * (s_x**2)) - (63 * s_x) + 305
    
    f_prime = diff(eq, s_x)
    f_prime_2 = diff(f_prime, s_x)
    
    if f_prime.subs(s_x, a) * f_prime_2.subs(s_x, a) < 0: 
        xk = a
        x[0] = b
    elif f_prime.subs(s_x, a) * f_prime_2.subs(s_x, a) > 0:
        xk = b
        x[0] = a
    else:
        return 0, []
    
    cieciwy[0] = np.array([[xk, x[0]], [f(xk), f(x[0])]], dtype=float)

    for i in range(1, n):
        x[i] = x[i-1] - f(x[i-1]) * (xk - x[i-1]) / (f(xk) - f(x[i-1]))

        cieciwy[i] = np.array([[xk, x[i]], [f(xk), f(x[i])]], dtype=float)

    return x[n-1], cieciwy
    



a = 10.510
b = 17.105

m = 10

k = 500
x = np.linspace(a, b, k)
y = np.zeros(k, dtype=float)
for i in range(k):
    y[i] = f(x[i])

x0_bisekcji, x_bisekcji_kroki, y_bisekcji_kroki = metoda_bisekcji(a, b, m)
x0_cieciw, cieciwy = metoda_cieciw(a, b, m)
x0_dokladne = 13.4297326385412000

print()

print(f"bisekcji: {x0_bisekcji:.16f}")
print(f"roznica z rozwiazaniem dokladnym: {(x0_bisekcji-x0_dokladne):.16f}")


plt.figure()
for i in range(x_bisekcji_kroki.shape[0]):
    plt.scatter(x_bisekcji_kroki[i], y_bisekcji_kroki[i],  marker='o', label=f'x{i+1} (krok {i+1})')
plt.scatter(x0_dokladne, 0,s=100, marker='o', color='red', label='x0')
plt.plot(x, y, color='blue', label='f(x) = f(x)=3x^2-63x+305')
plt.title("Metoda Bisekcji")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


print()
print()

print(f"cieciw: {x0_cieciw}")
print(f"roznica z rozwiazaniem dokladnym: {(x0_cieciw-x0_dokladne):.16f}")


plt.figure()
for i in range(cieciwy.shape[0]):
    plt.plot([cieciwy[i][0][0], cieciwy[i][0][1]], [cieciwy[i][1][0], cieciwy[i][1][1]],  marker='o', label=f'cieciwa (kroki {i+1})')
plt.scatter(x0_dokladne, 0, s=100, marker='o', color='red', label='x0')
plt.plot(x, y, color='blue', label='f(x) = f(x)=3x^2-63x+305')
plt.title("Metoda Cieciw")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()


