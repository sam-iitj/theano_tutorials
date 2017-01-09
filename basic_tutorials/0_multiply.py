import theano 
import theano.tensor as T

x = T.scalar('x')
y = T.scalar('y')

w = x * y 

f = theano.function(inputs=[x, y], outputs=w)

print f(1, 2)
print f(10, 20)
