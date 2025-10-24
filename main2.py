import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff


def f(x):
    return (3 * (x**2)) - (63 * x) + 305


def f_prime(a):
    s_x = symbols('s_x')
    eq = (3 * (s_x**2)) - (63 * s_x) + 305
    f_prime = diff(eq, s_x)
    return f_prime.subs(s_x, a)

def f_prime2(a):
    s_x = symbols('s_x')
    eq = (3 * (s_x**2)) - (63 * s_x) + 305
    f_prime = diff(eq, s_x)
    f_prime2 = diff(f_prime, s_x)
    return f_prime2.subs(s_x, a)


def wybor_poczatku(a, b):
    if f_prime(a) * f_prime2(a) < 0:
        return a 
    elif f_prime(a) * f_prime2(a) > 0:
        return b
    else:
        return 0




def metoda_newtona(a, b, n):
    x = np.zeros(n, dtype=float)

    x[0] = wybor_poczatku(a, b)

    for i in range(1, n):
        x[i] = x[i-1] - (f(x[i-1]) / f_prime(x[i-1]))

    return x[n-1], x

    


def metoda_newtona_raphsona(a, b, n):
    x = np.zeros(n, dtype=float)

    x[0] = wybor_poczatku(a, b)

    p = f_prime(x[0])
    for i in range(1, n):
        x[i] = x[i-1] - (f(x[i-1]) / p)

    return x[n-1], x



a = 10.510
b = 17.105

n = 10

k = 5000
x = np.linspace(a, b, k)
y = np.zeros(k, dtype=float)
for i in range(k):
    y[i] = f(x[i])

x0_dokladne = 13.4297326385412000



x0_newtona, x_kroki_newtona = metoda_newtona(a, b, n)

print()
print()

print(f"metoda newtona: {x0_newtona}")
print(f"roznica z rozwiazaniem dokladnym: {(x0_newtona-x0_dokladne):.16f}")

plt.figure(figsize=(8,8))
for i in range(1, x_kroki_newtona.shape[0]):
    plt.plot([x_kroki_newtona[i-1], x_kroki_newtona[i]], [f(x_kroki_newtona[i-1]), 0], '--',   marker='o', label=f'x{i} (krok {i})')
plt.scatter(x0_dokladne, 0, marker='o', color='black', label='x0')
plt.plot(x, y, color='blue', label='f(x) = f(x)=3x^2-63x+305')
plt.title("Metoda Newtona")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()





x0_newtona_raphsona, x_kroki_newtona_raphsona = metoda_newtona_raphsona(a, b, n)


print()
print()

print(f"metoda newtona: {x0_newtona_raphsona}")
print(f"roznica z rozwiazaniem dokladnym: {(x0_newtona_raphsona-x0_dokladne):.16f}")

plt.figure(figsize=(8,8))
for i in range(1, x_kroki_newtona_raphsona.shape[0]):
    plt.plot([x_kroki_newtona_raphsona[i-1], x_kroki_newtona_raphsona[i]], [f(x_kroki_newtona_raphsona[i-1]), 0], '--',  marker='o', label=f'x{i} (krok {i})')
plt.scatter(x0_dokladne, 0, marker='o', color='black', label='x0')
plt.plot(x, y, color='blue', label='f(x) = f(x)=3x^2-63x+305')
plt.title("Metoda Newtona-Raphsona")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
