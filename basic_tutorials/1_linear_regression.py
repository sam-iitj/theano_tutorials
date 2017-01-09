import theano 
import theano.tensor as T 
import numpy as np 
import matplotlib.pyplot as plt

trX = np.linspace(1, 10, 100)
trY = trX**4 + np.random.randn(*trX.shape) * 0.678

X = T.scalar()
Y = T.scalar()

def model(X, b, w1, w2):
    return b + X * w1 + (X**2) * w2

w1 = theano.shared(0., name='w1')
w2 = theano.shared(0., name='w2')
b  = theano.shared(0., name='b')

print b.get_value()
print w1.get_value()
print w2.get_value()

y = model(X, b, w1, w2)

cost = T.mean(T.sqr(y - Y))
gb, gw1, gw2 = T.grad(cost=cost, wrt=[b, w1, w2])
updates = [[b, b - gb * 0.1], [w1, w1 - w1 * gw1 * 0.1], [w2, w2 - w2 * gw2 * 0.1]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)

print b.get_value()
print w1.get_value()
print w2.get_value()

plt.scatter(trX, trY)
plt.plot(trX, model(trX, b.get_value(), w1.get_value(), w2.get_value()))
plt.show()
