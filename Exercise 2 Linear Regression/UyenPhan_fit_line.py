# Main
import matplotlib.pyplot as plt
import numpy as np


# Linear solver
def my_linfit(x, y):
    # Total number of training sets
    N = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Initialize numerator and denominator
    numerator = 0
    denominator = 0

    # For y = ax + b, calculate a
    for i in range(N):
        numerator = np.sum(x[i] * (y[i] - y_mean))
        denominator = np.sum(x[i] * (x[i] - x_mean))
    a = numerator / denominator

    # Calculate b
    b = y_mean - a * x_mean
    return a, b


x = []
y = []


def onclick(event):
    if event.button == 1:
        x.append(event.xdata)
        y.append(event.ydata)
        plt.plot(event.xdata, event.ydata, 'kx')
        plt.draw()  # redraw

    elif event.button == 3:  # Right click to stop collecting points
        plt.disconnect(cid)  # Disconnect the mouse click event handler
        a, b = my_linfit(x, y)
        xp = np.arange(-5, 10, 0.2)
        plt.plot(xp, a * xp + b, 'r-')
        plt.show()
        plt.draw()  # redraw
        print(f"My fit : a={a} and b={b}")


# fig, ax = plt.subplots()
# ax.scatter(x, y)
# fig.canvas.mpl_connect('button_press_event', onclick)

# x = np.random.uniform(-2, 5, 10)
# y = np.random.uniform(0, 3, 10)
plt.xlim([-5, 10])
plt.ylim([-5, 10])
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()
