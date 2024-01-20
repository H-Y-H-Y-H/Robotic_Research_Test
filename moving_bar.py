import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

# Create a figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.5)

# Some initial data

initial_amplitude = [0.5]*6

bars = plt.bar(range(1, 7), initial_amplitude)
plt.ylim(0,1)

# Position of the vertical slider
axcolor = 'lightgoldenrodyellow'

# Vertical Slider
amp_slider_list = []
for i in range(6):
    axamp = plt.axes([ 0.05 + 0.07 * i, 0.1, 0.02, 0.8], facecolor=axcolor)
    amp_slider_list.append(Slider(axamp, f'm{i}', 0., 1.0, valinit=initial_amplitude[i], orientation="vertical"))


# Update function
def update(val):
    for i in range(6):
        amplitude = amp_slider_list[i].val
        bars[i].set_height(amplitude)
    fig.canvas.draw_idle()

# Register the update function with the slider
for i in range(6):
    amp_slider_list[i].on_changed(update)

# Show the plot
plt.show()



