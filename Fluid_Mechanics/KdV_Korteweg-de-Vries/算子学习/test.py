import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)
y = -0.22/(np.cosh(x))
y2 = 0.22*(np.cosh(x))
# y2 = -0.3/(np.cosh(x)**2)
# plt.plot(x,y)
plt.plot(x,y2)
plt.savefig("cosh.png")
