# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Visualisation using
#
# ![Matplotlib](http://upload.wikimedia.org/wikipedia/en/5/56/Matplotlib_logo.svg)

# <a id=setup></a>
# # Notebook Setup (run me first!)

# First, we apply a "magic command" to make plots appear "inline" inside the notebook

# +
# matplotlib inline plotting
# # %matplotlib inline

# There's also a backend for interactive exploration
# %matplotlib ipympl

# +
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from pathlib import Path

# Make the size and fonts larger for this presentation
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2
# -

# <a id=line_plots></a>
# # Line Plots

# +
x = np.linspace(0, 1, 101) # 101 numbers from 0 to 1

plt.plot(x, x**2)
# -

# <a id=different_styles></a>
# ## Using different styles for plots

t = np.linspace(0, 2 * np.pi)   # 50 points between 0 and 2π
plt.plot(t, np.sin(t));

plt.plot(t, np.sin(t), '--');

plt.plot(t, np.sin(t), 'go')
# plt.plot(t, np.sin(t), color='green', marker='o', linestyle='');   # same thing!

# new in matplotlib 2.0, all colors of the color rotation available as C<N>
# Multiple lines in the same plot
x = np.linspace(0, 1, 100)
for n in range(9):
    plt.plot(x**(n + 1), color='C{}'.format(n))

# All styles and colors: [matplotlib.axes.Axes.plot](http://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html#matplotlib.axes.Axes.plot)
#
#

months = range(1, 13)
min_temp = np.array([-30.6, -34.3, -26.7, -11.7, -6.1, 1.1, 3.9, 3.3, -2.1, -8.2, -18.2, -22.8])
max_temp = np.array([21.7, 19.4, 31.7, 30.6, 36.7, 40.6, 40.6, 37.2, 37.2, 32.8, 26.1, 20.0])

# +
plt.plot(months, min_temp)#, c='0.2')
plt.plot(months, max_temp)#, color='0.2')
#plt.ylabel('$^\circ$C')

#plt.fill_between(months, min_temp, max_temp, color='lightgray')
#plt.grid(linestyle='--')
'''
month_names = ['January',
          'February',
          'March',
          'April',
          'May',
          'June',
          'July',
          'August',
          'September',
          'October',
          'November',
          'December']

plt.xticks(
    months,
    month_names,
    rotation=45,
    rotation_mode='anchor',
    horizontalalignment='right',  # or ha
    verticalalignment='top',      # or va
);
'''
# -

# <a id=histograms></a>
# # Histograms

# <a id=histogram_1d></a>
# ## 1D

# +
# plt.hist?

# +
sample_size = 100
m = 20
s = 5
normal_sample = np.random.normal(m, s, sample_size)
bins = np.linspace(0, 50, 101)
counts, bins, _ = plt.hist(normal_sample, bins=20, label='Random sample')
#plt.xlabel('Normally distributed variable?')
#plt.ylabel('Counts')
#plt.title('Normal distribution')

# Add a line plot to compare with expected distribution
#x = np.linspace(0, 40, 100)
#y = 1/np.sqrt(1*np.pi*s**2)*np.exp(-(x - m)**2/(2*s**2))*sample_size
#plt.plot(x, y, label=r'$p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }} e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} }$')

# Add a legend
#plt.legend(loc='upper left')
#plt.legend(loc=(0.2, 0.2))
# -

# <a id=histogram_2d></a>
# ## 2D

# +
# plt.hist2d?

# +
mean = [2, 1]
cov = [[9, 2],
       [2, 4]]

x, y = np.random.multivariate_normal(mean, cov, size=10000).T

hist_output = plt.hist2d(x, y, bins=50, cmap='gist_heat') # try different color maps: viridis(default), inferno, gist_heat
#plt.hist2d(x, y, bins=50)
# plt.hist2d(x, y, bins=[25, 50], range=[[-10, 14], [-5, 7]])

plt.colorbar(label='Counts'); 
# -

# <a id=bar></a>
# # Bar plots

# <a id=simple_bar></a>
# ## Simple bar plot

# +
# plt.bar?

# +
# Reusing the data from the 1D histogram
centers = (bins[:-1] + bins[1:])/2.
plt.bar(centers, counts)

# Not exactly the plot we had previously
# We have to set the width of the bar to the width of the bin
#bin_width = bins[1:] - bins[:-1]
#plt.bar(centers, counts, bin_width)
# -

# <a id=multiple_bar></a>
# ## Multiple bar plot

power_data = np.loadtxt('resources/power_sources.csv', delimiter=';', skiprows=1, usecols=range(1, 9))
power_headers = ['Hydroelectric',
 'Nuclear',
 'Coal',
 'Combined Cycle',
 'Wind',
 'Thermal',
 'Other non renewable',
 'Other renewable']
power_labels = ['January',
 'February',
 'March',
 'April',
 'May',
 'June',
 'July',
 'August',
 'September',
 'October',
 'November',
 'December']

plt.bar(np.arange(1, 13) - 1/4., power_data[:, 0], width=1/4., label=power_headers[0])
plt.bar(np.arange(1, 13), power_data[:, 4], width=1/4., label=power_headers[4])
plt.bar(np.arange(1, 13) + 1/4., power_data[:, 7], width=1/4., label=power_headers[7])
plt.xticks(np.arange(1, 13), power_labels, rotation=90)
plt.title('% of total Power produced in Spain in 2017')
plt.ylabel('%')
plt.legend()

# <a id=stacked_bar></a>
# ## Stacked bar plot

# +
renewable = power_data[:, [0, 4, 7]].sum(axis=1) # Hydroelectric + Wind + Other renewable'
non_renewable = power_data[:, [1, 2, 3, 5, 6]].sum(axis=1) # Nuclear + Coal + Combined Cycle + Thermal + Other non renewable

plt.bar(range(1, 13), renewable, label='Renewable')
plt.bar(range(1, 13), non_renewable, bottom=renewable, label='Non renewable')

plt.xticks(range(1, 13), power_labels, rotation=45, ha='right')
plt.hlines(50, 0, 13, colors='k')
plt.legend()

#plt.yticks([]);
# -

# <a id=scatter></a>
# # Scatter plots

# +
# plt.scatter?

# +
x1, y1, z1 = np.random.multivariate_normal([1, 1, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 10]], 1000).T

# raw plot
plt.scatter(x1, y1, alpha=0.5)

# Coloured
#plt.scatter(x1, y1, c=z1, cmap='winter')
#plt.colorbar()

# With sizes
#plt.scatter(x1, y1, s=5*(z1 - min(z1)), alpha=0.5)

# Add another scatter
#x2, y2 = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 1000).T
#plt.scatter(x2, y2, alpha=0.5)

# Add a grid
#plt.grid()
# And tune it!!
#plt.grid(alpha=0.2, linestyle='--', linewidth=2., color='0.2')

# The grid is bound to the ticks
#plt.xticks(np.arange(-4, 4.1))
#plt.yticks(np.arange(-4, 4.1));

# Set the limits of the plot
#plt.xlim([-2, 2])
#plt.ylim([-2, 2]);

# -

# <a id=exercise_1></a>
# # Exercise 1
#
# Generate a line plot with the data of 3 random walks, corresponding to different random distributions.
#
# With these characteristics:
# * All lines in grey tones
# * different markers or line types for each line
# * with a legend located at the 'upper left'
#
# **Hint:** Use the `np.cumsum` function to generate a random walk from a random sample.  

# <a id=architecture></a>
# # matplotlib architecture
#
# Up to this point, we have only used the so-called *state-machine interface*. It's a MATLAB-like interface implemented in the *pyplot* module that provides convenient methods for easy plotting. But it has some drawbacks:
#
# * It's not very pythonic (python is OO)
# * No idea what's happening in the background
#
# This interface is OK for online plotting, because it's simple, but it is not convenient for other purposes, like embedding plots into a program. For all other purposes, the OO interface should be preferred. Using it, our code will be more verbose, but also more explanatory, and we'll have more control on what it's doing.
#
#
# In order to introduce the OO API, it's convenient that we have some knowledge of matplotlib's architecture.
#
# The top-level matplotlib object that contains and manages all of the elements in a given graphic is called the **Figure**.
#
# To achieve the manipulation and rendering of this objects, matplotlib is structured in three layers:
# * *Backend*: It's the one that actually draws the Artists on the canvas.
# * *Artist Layer*: Are the things that have to be plotted. The figure, lines, axis, bars, ...
# * *Scripting Layer (pyplot)*: Light scripting interface we have alredy shown
#
#
# ## Artists layer
#
# Everything you see in a plot is an Artist instance. This layer is a hierarchy of objects with the **Figure** sitting on top of it.
#
# <table><tr>
# <td><img src="resources/artists_figure.png"></img></td>
# <td><img src="resources/artists_tree.png"></img></td>
# </tr>
# </table>
#
#
# Some more detail ([directly from the matplotlib documentation](https://matplotlib.org/tutorials/introductory/usage.html#parts-of-a-figure))
#
# ### Figure
# The whole figure. The figure keeps track of all the child Axes, a smattering of 'special' artists (titles, figure legends, etc), and the canvas. A figure can have any number of Axes.
#
# ### Axes
# This is what you think of as 'a plot', it is the region of the image with the data space. A given figure can contain many Axes, but a given Axes object can only be in one Figure. The Axes contains two (or three in the case of 3D) Axis objects.
#
# The Axes class and it's member functions are the primary entry point to working with the OO interface.
#
# ### Axis
# These are the number-line-like objects. They take care of setting the graph limits and generating the ticks (the marks on the axis) and ticklabels (strings labeling the ticks).

# +
x = np.linspace(0, 10, 100)
y = x**2
plt.plot(x, y, label='ay')
plt.xlabel('lolay')
plt.ylabel('lolá')
plt.legend()

# import the artist class from matplotlib
from matplotlib.artist import Artist

def rec_gc(art, depth=0, max_depth=8):
    if depth < max_depth and isinstance(art, Artist):
        # increase the depth for pretty printing
        print("  " * depth + str(art))
        for child in art.get_children():
            rec_gc(child, depth+2)

# Call this function on the legend artist to see what the legend is made up of
fig = plt.gcf()
rec_gc(fig)
# -

# Both APIs can be connected through the `pyplot.gcf` and `pyplot.gca` methods.

# +
x = np.arange(5)
plt.plot(x, x)

fig = plt.gcf()
ax = plt.gca()
print('Is this the same? {}\n'.format(ax == fig.get_children()[1]))
# -

plt.get_backend()

# When using the inline backend, the current figure is renderd in cell execution.
#
# New cells will contain new figures.
#
# This behaviour may be different with other backends.

new_fig = plt.gcf()
print('old figure exits and is a {}'.format(type(fig)))
print('new figure exits and is a {}'.format(type(new_fig)))
print('But are they the same instance? {}'.format(fig == new_fig))

# <a id=multiple_plots></a>
# # Multiple plots in the same figure

# +
x = np.linspace(0, 3.)

# subplot arguments: # of rows, # of columns, plot index (row * (#cols) + col)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1) # the same as fig.add_subplot
    ax.plot(x, x**i)
    ax.set_xlim(0, 3)
    y_center = np.array(ax.get_ylim()).sum()/2.
    ax.text(1.5, y_center, str(i + 1), ha='center', va='center', fontsize=32)

plt.tight_layout() # When doing multiple plots you should almost always use this command


# -

# <a id=shared_axis></a>
# ## Subplots / Shared Axes

# +
def poisson(x, k):
    return np.exp(-x)*x**k / np.math.factorial(k)

x = np.linspace(0, 12, 40)
y = poisson(x, 2)
y_noise = y + np.random.normal(0, 0.01, len(y))
z = np.linspace(0, 12, 100)

gridspec = {'height_ratios': [2, 1]}
fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw=gridspec)

ax1.plot(x, y_noise, 'ko')
ax1.plot(z, poisson(z, 2))
ax1.set_ylim(-0.05, 0.30)
ax1.set_ylabel('Flux')
#ax1.set_yticks(ax1.get_yticks()[1:])    # remove bottom y-tick

ax2.plot(x, y_noise - y, 'ko')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.set_xlabel('Energy')
ax2.set_ylim(-0.03, 0.04)
ax2.set_ylabel('Residuals')
#ax2.set_yticks(ax2.get_yticks()[:-2])   # remove top y-tick

#fig.subplots_adjust(hspace=0)
fig.suptitle('\nFake Spectrum', fontweight='bold');
# -

# <a id=gridspec></a>
# ## Using Gridspec

# +
# use gridspec to partition the figure into subplots
import matplotlib.gridspec as gridspec

fig = plt.figure()
gspec = gridspec.GridSpec(3, 3) #, wspace=0.05, hspace=0.05)

top_histogram = fig.add_subplot(gspec[0, 1:])
side_histogram = fig.add_subplot(gspec[1:, 0])
lower_right = fig.add_subplot(gspec[1:, 1:])

# We produce two sets of random data
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)

# Normed histograms counts add up to 1, they resemble a probability density function
top_histogram.hist(X, bins=100, density=True)
side_histogram.hist(Y, bins=100, orientation='horizontal', density=True)
# flip the side histogram's x axis
#side_histogram.invert_xaxis()
lower_right.scatter(X, Y, alpha=0.5)

# Remove the picks in-between
#top_histogram.set_xticks([])
#lower_right.set_yticks([])
#top_histogram.set_yticks(top_histogram.get_yticks()[1:])
#side_histogram.set_xticks(side_histogram.get_xticks()[1:]);
# -

# <a id=inset_plots></a>
# ## Inset Plots (plot inside a plot)

# +
plt.plot(x, x**2)
plt.title("Outer Plot")

# axes coordinates: (0,0) is lower left, (1,1) upper right
plt.axes([0.2, 0.45, 0.3, 0.3])
plt.plot(x, x**3)
plt.title("Inner Plot");
# -

# <a id=exercise_2></a>
# ## Exercise 2
#
# Generate 3 differents random samples.
#
# Plot them as normalized 1D histograms in a figure with 3 axes in 1 column, with shared x-axis and y-axis.
#
# Put the name of the distribution somewhere in the axes so it can be identified.
#
# There have to be no x-ticks in the top 2 axes and no vertical space between them.
#
# Remove y-ticks.

# <a id=other_stuff></a>
# # Other stuff

# <a id=plot_image></a>
# ##  Plot image

n_rows = 2
n_cols = 2
m = np.array([[n_rows*i + j for j in range(n_cols)] for i in range(n_rows)])
plt.imshow(m)
plt.colorbar();

# +
from matplotlib.colors import LogNorm

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 120).reshape(-1, 1)

im = plt.imshow(np.sin(x) + np.cos(y), cmap='gist_heat') #, clim=(-0.5, 0.5)) #, norm=LogNorm())
plt.colorbar();
# -

# ## Shape plots

# +
from matplotlib.patches import Circle, Ellipse, Rectangle, Polygon
from matplotlib.collections import PatchCollection

# Fixing random state for reproducibility
np.random.seed(19680801)


kwargs1 = {}
kwargs2 = {'facecolor': 'None',
          'edgecolor': 'tab:red',
          'linewidth': 5}
kwargs3 = {'hatch': '*',
          'fc': 'None'}
kwargs3 = {'fc': 'greenyellow',
          'alpha': 0.5}

fig, ax = plt.subplots(figsize=(20, 10))

# Rectangles: all patches have the same style 
patch_list = []
hr = 30
wr = -20
patch_list.append(Rectangle((wr + 2, hr), 1, 1))
patch_list.append(Rectangle((wr + 6, hr), 2, 0.5))
patch_list.append(Rectangle((wr + 12, hr), 2, 4))
p = PatchCollection(patch_list, **kwargs1)
ax.add_collection(p)

# Polygon: all patches have the same style
patch_list = []
hr = 23
wr = -20
xor = [wr + i for i in [2, 6, 12]]
radius = [1, 2, 3]
for ind, nside in enumerate([3, 5, 7]):
    points = [(xor[ind] + radius[ind]*np.cos(2*np.pi*i/nside),
               hr + radius[ind]*np.sin(2*np.pi*i/nside))
               for i in range(nside)]
    patch_list.append(Polygon(points))
p = PatchCollection(patch_list, **kwargs2)
ax.add_collection(p)


# Ellipse: all patches have the same style 
patch_list = []
hel = 30
we = 0
patch_list.append(Ellipse((we + 2, hel), 1, 2))
patch_list.append(Ellipse([we + 6, hel], 2, 0.5))
patch_list.append(Ellipse([we + 12, hel], 2, 4, angle=45))
p = PatchCollection(patch_list, **kwargs3)
ax.add_collection(p)

# Circle: each patch is independent
hc = 23
wc = 0
ax.add_patch(Circle([wc + 2, hc], 1, **kwargs1))
ax.add_patch(Circle([wc + 6, hc], 2, **kwargs2))
ax.add_patch(Circle([wc + 12, hc], 3, **kwargs3))

ax.set_xlim(-20, 16)
ax.set_ylim(18, 36)
ax.set_aspect('equal')
    
# -

# <a id=box_plot></a>
# ## Box plots

s1 = np.random.normal(size=100)
s2 = np.random.uniform(size=100)
s3 = np.random.exponential(size=100)
plt.boxplot([s1, s2, s3], labels=['Normal', 'Uniform', 'Exponential']);

# <a id=pie_chart></a>
# ## Pie charts

# +
pie_order = [0, 4, 7, 1, 2, 3, 5, 6]
pie_data = power_data[11, pie_order]
pie_labels = [power_headers[i] for i in pie_order]
pie_colors = [(0, (i + 3)/5., 0, 0.7) for i in range(3)] + \
    [((i + 3)/7., 0, 0, 0.7) for i in range(5)] #RGB color specification

fig, ax = plt.subplots()
ax.pie(pie_data, labels=pie_labels, colors=pie_colors)
ax.axis('equal');
# -

# <a id=projections></a>
# ## Projections

# +
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(1, 3, wspace=0.5)

theta = np.linspace(0, 5*2*np.pi, 101)
r = theta/2*np.pi

fig = plt.figure(figsize=(16, 8))

ax1 = fig.add_subplot(gs[0], projection='polar')
ax1.plot(theta, r)

gaia_data = np.loadtxt('resources/GaiaDR2.csv', delimiter=',', skiprows=11)
ax2 = fig.add_subplot(gs[1:], projection='mollweide')
ax2.scatter(np.deg2rad(gaia_data[:, 0]), np.deg2rad(gaia_data[:, 1]), alpha=0.3)
ax2.grid()

# -

# <a id=plot_styles></a>
# ## Plot styles

# List available styles:

print(plt.style.available)

# +
from scipy import stats

def plot_stuff():
    plt.subplot(2, 2, 1)
    x = np.linspace(-1, 1, 1000)
    plt.plot(x, np.sin(50*x**3)/(x))
    plt.grid()

    plt.subplot(2, 2, 2)
    x = np.linspace(-1, 1, 10)
    y = np.exp(-2.2*x) + np.random.normal(0, 0.1, 10)
    yerr = np.abs(np.random.normal(0, 0.2, 10))
    plt.errorbar(x, y, yerr, fmt='o', capsize=3)
    plt.yscale('log')

    plt.subplot(2, 2, 3)
    x = stats.skewnorm.rvs(10, size=1000)
    plt.hist(x, bins=50)

    plt.subplot(2, 2, 4)
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((x, y))
    z = stats.multivariate_normal([0.1, 0.3], [[0.2, 0.3], [0.1, 0.4]])
    plt.contourf(x, y, z.pdf(pos))

for plot_style in ['classic', 'bmh', 'fivethirtyeight', 'ggplot', 'seaborn']:
    plt.figure()
    with plt.style.context(plot_style):   # use context manager so that changes are temporary
        plot_stuff()
        plt.suptitle('Plot Style: ' + plot_style, fontweight='bold')
# -

# <a id=save_figures></a>
# ## Saving figures
#
# Use `plt.savefig` to save your figure.
#
# You can either give path relative to your working directory or an absolute path.
# Not sure what the current working directory is?

pwd()

# +
x = np.linspace(-5, 5)
plt.plot(x, x**3, marker='s')
plt.title("My Awesome Plot")
plot_folder = Path('plots')

# save in dedicated directory; extension determines file type
plt.savefig(plot_folder / 'awesome_plot.pdf')
plt.savefig(plot_folder / 'awesome_plot.eps')
plt.savefig(plot_folder / 'awesome_plot.png', dpi=300)
plt.savefig(plot_folder / 'awesome_plot.jpg', dpi=300)

# absolute path
# plt.savefig('/path/to/output/directory/awesome_plot.pdf')
# -

# <a id=animations></a>
# ## Animations

# +
import matplotlib.animation as animation

class FunctionAnimatedImage():
    
    def func(self):
        return np.sin(self.x) + np.cos(self.y)
    
    def __init__(self):
        self.x = np.linspace(0, 2 * np.pi, 120)
        self.y = np.linspace(0, 2 * np.pi, 120).reshape(-1, 1)

        self.im = plt.imshow(self.func(), animated=True)
        
    def next_frame(self, i, *args):
        
        self.x += np.pi / 5.
        self.y += np.pi / 20.
        self.im.set_array(self.func())
        return self.im,

fig = plt.figure()
anim_img = FunctionAnimatedImage()

# Animate the image by recursively calling the next_frame function
ani_1 = animation.FuncAnimation(fig, anim_img.next_frame, frames=40, interval=50, blit=True)

# Embed the video in an html 5.0 video tag
HTML(ani_1.to_html5_video())
# -

# <a id=3d_plots></a>
# ## 3D Plots

# +
import mpl_toolkits.mplot3d.axes3d as p3
x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 120).reshape(-1, 1)
z = np.sin(x) + np.cos(y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
# -

# # Other plotting libraries
#
# ## Seaborn
#
# High-level plotting library on top of maplotlib

import seaborn as sns

# +
np.random.seed(1234)

v1 = np.random.normal(0, 10, 1000)
v2 = 2*v1 + np.random.normal(60, 15, 1000)
# -

# plot a kernel density estimation over a stacked barchart
plt.figure()
plt.hist([v1, v2], histtype='barstacked', density=True);
v3 = np.concatenate((v1,v2))
sns.kdeplot(v3);

grid = sns.jointplot(x=v1, y=v2, alpha=0.4);
#grid.ax_joint.set_aspect('equal')

# And much more ... https://seaborn.pydata.org/

# + [markdown] toc-hr-collapsed=true
# ## Bokeh 
#
# Python library to produce interactive plots
#
# https://docs.bokeh.org/en/latest/index.html
# -

import numpy as np
from bokeh.io import push_notebook, show, output_notebook
from bokeh.models import HoverTool
from bokeh.plotting import figure 
output_notebook()

# +
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = np.array([(r, g, 150) for r, g in zip(50+2*x, 30+2*y)], dtype="uint8")

TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,examine,help"

p = figure(tools=TOOLS)

p.scatter(x, y, radius=radii,
          fill_color=colors, fill_alpha=0.6,
          line_color=None)

show(p, notebook_handle=True)
# -

# ## plotly
#
# https://plotly.com/python/

import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color='petal_length')
fig.show()

# # Interactive plotting

# ## with ipympl
#
# By changing the backend ( %matplotlib inline -> %matplotlib ipympl ) . We will already have interactive exploration for free.
#
# Change the backend, restart the notebook and regenerate any of the previous plots
#
# The data in the plots can be dynamically updated.

# +
# One can bound figure attributes to other widget values.
from ipywidgets import AppLayout, FloatSlider

plt.ioff()

slider = FloatSlider(
    orientation='horizontal',
    description='Factor:',
    value=1.0,
    min=0.02,
    max=2.0
)

slider.layout.margin = '0px 30% 0px 30%'
slider.layout.width = '40%'

fig = plt.figure()
fig.canvas.header_visible = False
fig.canvas.layout.min_height = '400px'
plt.title('Plotting: y=sin({} * x)'.format(slider.value))

x = np.linspace(0, 20, 500)

lines = plt.plot(x, np.sin(slider.value * x))

def update_lines(change):
    plt.title('Plotting: y=sin({} * x)'.format(change.new))
    lines[0].set_data(x, np.sin(change.new * x))
    fig.canvas.draw()
    fig.canvas.flush_events()

slider.observe(update_lines, names='value')

AppLayout(
    center=fig.canvas,
    footer=slider,
    pane_heights=[0, 6, 1]
)
# -

# <a id=exercise_3></a>
# # Exercise 3
#
# Load the file `resources/nip.fits` using fits.
# ```
# hdul = fits.open('resources/nip.fits')
# ```
#
# Plot the matrix in `hdul[1].data` with imshow. You won't see much.
#
# Do a histogram (Hint: use the `log` argument) to analyze the values in the matrix.
#
# Use imshow (with `norm`) to see what's inside the plot.
#
# Use `imshow` + `clim` to see a higher detail of the background.
#
# Put everything in a single figure mith multiple axes

from astropy.io import fits

hdul = fits.open('resources/nip.fits')


