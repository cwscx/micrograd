# Micrograd is a small version of autoGrad engine, which stands for automatic gradient.
# Automatic gradient engine implements the backpropagation algorithms, to 
# iteratively tune the neural network to minimize some kind of loss function.

from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b ** 3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e ** 2
g = f / 2.0
g += 10.0 / f

## Gradient is derivative of 'g' respect to derivative of 'a'
## It is telling how node are affecting the final output. e.g.
## How 'a' is affecting 'g'.
print(f'g data: {g.data:.4f}')
print(f'g grad: {g.grad:.4f}') # 1.0
print(f'b grad: {b.grad:.4f}') # 
g.backward()
print(f'g grad: {a.grad:.4f}')
print(f'b grad: {b.grad:.4f}')  # Tell us 