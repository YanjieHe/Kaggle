import matplotlib.pyplot as plt 
import pandas as pd
import math

def array_to_image(array):
    n = array.shape[0]
    n = int(math.sqrt(n))
    pixels = array.reshape((n, n))
    return pixels

def plot_array_image(array):
    pixels = array_to_image(a)
    plt.imshow(pixels, cmap = "gray")
    plt.show()

def plot_images(df, nrows, ncols):
    matrix = df.iloc[0: (nrows * ncols), 1:].values
    (figure, ax) = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            pixels = array_to_image(matrix[i * ncols + j])
            ax[i][j].imshow(pixels, aspect = "auto")
    plt.show()
