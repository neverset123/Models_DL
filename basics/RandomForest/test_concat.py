import numpy as np
import random

x=np.random.rand(4,2)
y=np.random.rand(4, 1)

print(x)
print(y)

combined=np.concatenate((x,y), axis=1)
print(np.shape(combined))
