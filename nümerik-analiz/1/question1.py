# Nagihan Baz - 171805024 - Question 1

import math

x = 2
e_to_2 = 0
for i in range(10):
    e_to_2 += x ** i / math.factorial(i)

print(math.exp(2))



def func_e(x, n):
    e_approx = 0
    for i in range(n):
        e_approx += x ** i / math.factorial(i)

    return e_approx
out = func_e(5,10)
print(out)

out = math.exp(5)
print(out)
x = 5
for i in range(1,6):
    e_approx = func_e(x,i)
    e_exp = math.exp(x)
    e_error = abs(e_approx - e_exp)
    print(f'{i} terms: Taylor Series approx= {e_approx}, exp calc= {e_exp}, error = {e_error}')


    def func_cos(x, n):
        cos_approx = 0
        for i in range(n):
            coef = (-1) ** i
            num = x ** (2 * i)
            denom = math.factorial(2 * i)
            cos_approx += (coef) * ((num) / (denom))

        return cos_approx


    angle_rad = (math.radians(45))
    out = func_cos(angle_rad, 5)
    print(out)
    out = math.cos(angle_rad)
    print(out)