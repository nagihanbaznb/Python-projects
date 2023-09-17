# Nagihan Baz - 171805024 - Question 4

f1 = lambda u, v, w: (-2 + v - w) / 3
f2 = lambda u, v, w: (-1 + u - 2 * w) / 8
f3 = lambda u, v, w: (4 - u - v) / 5

u0 = 0
v0 = 0
w0 = 0
count = 1

e = float(input('Enter tolerable error: '))

print('\nCount\tu\tv\tw\n')

condition = True

while condition:
    u1 = f1(u0, v0, w0)
    v1 = f2(u0, v0, w0)
    w1 = f3(u0, v0, w0)
    print('%d\t%0.4f\t%0.4f\t%0.4f\n' % (count, u1, v1, w1))
    e1 = abs(u0 - u1);
    e2 = abs(v0 - v1);
    e3 = abs(w0 - w1);

    count += 1
    u0 = u1
    v0 = v1
    w0 = w1

    condition = e1 > e and e2 > e and e3 > e

print('\nSolution: u=%0.3f, v=%0.3f and w = %0.3f\n' % (u1, v1, w1))