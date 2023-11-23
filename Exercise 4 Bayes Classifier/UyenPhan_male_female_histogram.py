import numpy as np
import matplotlib.pyplot as plt

X_train = np.loadtxt('male_female_X_train.txt')
y_train = np.loadtxt('male_female_y_train.txt')

# male and female heights (the first) and weights (the second column).
male_heights = []
male_weights = []

female_heights = []
female_weights = []

for i in range(len(y_train)):
    if y_train[i] == 0:
        male_heights.append(X_train[i, 0])  # Append only the height (first column)
        male_weights.append(X_train[i, 1])
    else:
        female_heights.append(X_train[i, 0])
        female_weights.append(X_train[i, 1])

# Create subplot for heights and weights histogram
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # (row, cols, figsize(width, length))
#
# # Histogram of heights of the 2 classes - 10 bins
# axes[0].hist(male_heights, bins=10, alpha=0.5, range=[80, 220], label='Male', color='cyan')
# axes[0].hist(female_heights, bins=10, alpha=0.5, range=[80, 220], label='Female', color='purple')
# axes[0].legend()
# axes[0].set_title('Height Histogram')

height_plot = plt.figure(1)
plt.hist(male_heights, bins=10, alpha=0.5, range=[80, 220], label='Male', color='cyan')
plt.hist(female_heights, bins=10, alpha=0.5, range=[80, 220], label='Female', color='purple')
plt.xlabel('Height (in cm)')
plt.ylabel('Number of people')
plt.title('Height Histogram')
plt.legend()
# axes[1].hist(male_weights, bins=10, alpha=0.5, range=[30, 180], label='Male', color='cyan')
# axes[1].hist(female_weights, bins=10, alpha=0.5, range=[30, 180], label='Female', color='purple')
# axes[1].legend()
# axes[1].set_title('Weight Histogram')
# plt.tight_layout()
# plt.show()

weight_plot = plt.figure(2)
plt.hist(male_weights, bins=10, alpha=0.5, range=[30, 180], label='Male', color='cyan')
plt.hist(female_weights, bins=10, alpha=0.5, range=[30, 180], label='Female', color='purple')
plt.xlabel('Weight (in kg)')
plt.ylabel('Number of people')
plt.title('Weight Histogram')
plt.legend()

plt.show()
