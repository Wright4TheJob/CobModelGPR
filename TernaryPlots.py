# https://github.com/marcharper/python-ternary
import ternary
import random
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

def random_points(num_points=25, scale=40):
    points = []
    for i in range(num_points):
        x = random.randint(1, scale)
        y = random.randint(0, scale - x)
        z = scale - x - y
        points.append((x,y,z))
    return points

def generate_random_heatmap_data(scale=5):
    from ternary.helpers import simplex_iterator
    d = dict()
    for (i,j,k) in simplex_iterator(scale):
        d[(i,j)] = random.random()
    return d

def generate_data(function,n):
    x = list(np.linspace(0.0, 1.0, num=n))
    y = list(np.linspace(0.0, 1.0, num=n))
    d = dict()
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            k = 1.0 - x[i] - y[j]
            inputs = [x[i],y[j],k]
            d[(i,j)] = function(inputs)
    return d

def heatmap(function,scale=100,title="Title"):
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 8)
    tax.heatmapf(function, boundary=True, style="triangular")
    tax.boundary(linewidth=2.0)
    tax.set_title(title)
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    fontsize = 20
    tax.left_axis_label("Straw", fontsize=fontsize)
    tax.right_axis_label("Clay", fontsize=fontsize)
    tax.bottom_axis_label("Sand", fontsize=fontsize)
    tax.clear_matplotlib_ticks()

    tax.show()

def heatmap_data(data,scale=100,title="Title"):
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(10, 8)
    tax.heatmap(data, scale=scale, style="triangular")
    tax.boundary(linewidth=2.0)
    tax.set_title(title)
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    fontsize = 20
    tax.left_axis_label("Straw", fontsize=fontsize)
    tax.right_axis_label("Clay", fontsize=fontsize)
    tax.bottom_axis_label("Sand", fontsize=fontsize)
    tax.clear_matplotlib_ticks()

    tax.show()

def scatter(points):
    ### Scatter Plot
    scale = 100
    figure, tax = ternary.figure(scale=scale)
    ax = tax.get_axes()
    tax.set_title("Scatter Plot", fontsize=20)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=5, color="blue")
    # point = (bottomAxis,rightAxis,leftAxis)
    p1 = (10,60,30)
    p2 = (70,0, 30,)
    tax.line(p1, p2, linewidth=3., marker='s', color='green', linestyle=":")
    tax.left_parallel_line(80, linewidth=2., color='red', linestyle="--")

    # Plot a few different styles with a legend
    points = random_points(30, scale=scale)
    tax.scatter(points, marker='s', color='red', label="Experimental Results")
    #tax.legend()
    tax.ticks(axis='lbr', linewidth=1, multiple=5)
    fontsize = 20
    tax.left_axis_label("Straw", fontsize=fontsize)
    tax.right_axis_label("Clay", fontsize=fontsize)
    tax.bottom_axis_label("Sand", fontsize=fontsize)
    #plt.ylim(ymin=-20)
    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()

    tax.show()
