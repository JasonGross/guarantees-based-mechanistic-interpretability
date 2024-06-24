# %%
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    print("Not in IPython, not loading autoreload")
# %%
import matplotlib.pyplot as plt
import numpy as np

# Create the data points
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Compute the color based on max(x, y)
color = np.maximum(xx, yy)

# Plot with gradient color
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(xx, yy, c=color, cmap="viridis")
plt.colorbar(sc1, label="max(x, y)")
plt.title("Gradient Color by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

# Plot with distinctive colors
plt.subplot(1, 2, 2)
unique_colors = np.unique(color)
cmap = plt.cm.get_cmap("tab20", len(unique_colors))
color_map = cmap(color)

sc2 = plt.scatter(xx, yy, c=color_map)
plt.title("Distinctive Colors by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()


# %%
import matplotlib.pyplot as plt
import numpy as np

# Create the data points
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Compute the color based on max(x, y)
color = np.maximum(xx, yy)

# Plot with gradient color
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(xx, yy, c=color, cmap="viridis")
plt.colorbar(sc1, label="max(x, y)")
plt.title("Gradient Color by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

# Plot with distinctive colors
plt.subplot(1, 2, 2)
unique_colors = np.unique(color)
cmap = plt.cm.get_cmap("tab20", len(unique_colors))
color_map = cmap(color)

sc2 = plt.scatter(xx, yy, c=color_map)
plt.title("Distinctive Colors by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Create the data points
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Compute the color based on max(x, y)
color = np.maximum(xx, yy)

# Plot with gradient color
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(xx, yy, c=color, cmap="viridis", s=10)  # s=10 to set the point size
plt.colorbar(sc1, label="max(x, y)")
plt.title("Gradient Color by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

# Plot with distinctive colors
plt.subplot(1, 2, 2)
unique_colors = np.unique(color)
cmap = plt.cm.get_cmap("tab20", len(unique_colors))
color_map = cmap(color)

sc2 = plt.scatter(xx, yy, c=color_map, s=10)  # s=10 to set the point size
plt.title("Distinctive Colors by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Create the data points
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Compute the color based on max(x, y)
color = np.maximum(xx, yy)

# Plot with gradient color using hsv
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(
    xx, yy, c=color % 32, cmap="hsv", s=10
)  # Using modulus to cycle through colors
plt.colorbar(sc1, label="max(x, y) % 32")
plt.title("Cyclic Gradient Color by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

# Plot with distinctive cycling colors
plt.subplot(1, 2, 2)
cmap = plt.cm.get_cmap("tab20", 20)  # Define a colormap with 20 colors
color_map = cmap(color % 20)

sc2 = plt.scatter(xx, yy, c=color_map, s=10)  # Using modulus to cycle through colors
plt.title("Distinctive Cycling Colors by max(x, y)")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Create the data points
x = np.arange(0, 64)
y = np.arange(0, 64)
xx, yy = np.meshgrid(x, y)
xx = xx.flatten()
yy = yy.flatten()

# Compute the color and alpha based on max(x, y) and min(x, y)
color = np.maximum(xx, yy)
alpha = 1 - (color - np.minimum(xx, yy)) / 63

# Plot with gradient color using hsv and adjusted alpha
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sc1 = plt.scatter(
    xx, yy, c=color % 32, cmap="hsv", s=10, alpha=alpha
)  # Using modulus to cycle through colors
plt.colorbar(sc1, label="max(x, y) % 32")
plt.title("Cyclic Gradient Color by max(x, y) with Alpha Fade")
plt.xlabel("x")
plt.ylabel("y")

# Plot with distinctive cycling colors and adjusted alpha
plt.subplot(1, 2, 2)
cmap = plt.cm.get_cmap("tab20", 20)  # Define a colormap with 20 colors
color_map = cmap(color % 20)

sc2 = plt.scatter(
    xx, yy, c=color_map, s=10, alpha=alpha
)  # Using modulus to cycle through colors
plt.title("Distinctive Cycling Colors by max(x, y) with Alpha Fade")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()

# %%
