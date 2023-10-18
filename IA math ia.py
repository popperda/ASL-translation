# importing libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# defining surface and axes
x = np.outer(np.linspace(-2, 2, 10), np.ones(10))
y = x.copy().T
z = (np.sqrt((1-x)*(1-y)*(x+y-1)))

fig = plt.figure()

# syntax for 3-D plotting
ax = plt.axes(projection='3d')

# syntax for plotting
ax.plot_surface(x, y, z, cmap='viridis',\
				edgecolor='green')
ax.set_title('Surface plot geeks for geeks')
plt.show()
