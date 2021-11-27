import matplotlib.pyplot as plt # Plotting utils
from mpl_toolkits.axes_grid1 import ImageGrid # More plotting...
import numpy as np; import math # Numpy and math

# Plots a single row of images
def __plot_image_row(data):
    num_plots = data.shape[0]
    fig = plt.figure(figsize=(num_plots, 10.*num_plots))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, num_plots), axes_pad=0.1)
    for i in range(num_plots):
        grid[i].imshow(data[i].reshape((56, 56)))
    plt.show()

# Plots a series of rows of images given a range
def plot_images(data, total_digit_range = None, row_size = 10):
    if (total_digit_range == None): total_digit_range = range(0, len(data))
    num_rows =  math.ceil(total_digit_range.stop / row_size)
    rows = [i * row_size for i in range(0, num_rows)]
    rows = [range(max(total_digit_range.start, start), min(start + row_size, total_digit_range.stop)) for start in rows]
    for row in rows: __plot_image_row(np.array([data[i] for i in row]))

# plot training vs. validation losses
def plot_losses(train_losses, valid_losses):
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')