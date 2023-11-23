import numpy as np

bins = 10
X_train = np.loadtxt('male_female_X_train.txt')
y_train = np.loadtxt('male_female_y_train.txt')
X_test = np.loadtxt('male_female_X_test.txt')
y_test = np.loadtxt('male_female_y_test.txt')

# Calculate class priors
num_male = np.sum(y_train == 0)
num_female = np.sum(y_train == 1)
prior_male = num_male / len(y_train)
prior_female = num_female / len(y_train)

# histograms for height and weight in the training data
height_hist, height_bins = np.histogram(X_train[:, 0], bins=bins, range=(80, 220))
weight_hist, weight_bins = np.histogram(X_train[:, 1], bins=bins, range=(30, 180))

# Calculate bin centroid
# height_bins[:-1] - exclude the last element
height_bin_centroids = (height_bins[:-1] + height_bins[1:]) / 2
weight_bin_centroids = (weight_bins[:-1] + weight_bins[1:]) / 2

# Initialize counters for males and females in each bin of height and weight
# can be zeros, but alpha = 1 to never get 0 for probabilities
males_in_height_bin = np.ones(bins)
males_in_weight_bin = np.ones(bins)
females_in_height_bin = np.ones(bins)
females_in_weight_bin = np.ones(bins)

# Iterate through the training data and count males and females in each bin
for i in range(len(X_train)):
    height = X_train[i, 0]
    weight = X_train[i, 1]
    gender = y_train[i]  # 0 for male and 1 for female

    # Find the closest bin distance for training data
    closest_height_bin = np.argmin(np.abs(height - height_bin_centroids))
    closest_weight_bin = np.argmin(np.abs(weight - weight_bin_centroids))

    # Increment the counters based on gender
    if gender == 0:  # Male
        males_in_height_bin[closest_height_bin] += 1
        males_in_weight_bin[closest_weight_bin] += 1
    else:  # Female
        females_in_height_bin[closest_height_bin] += 1
        females_in_weight_bin[closest_weight_bin] += 1

p_height_male = males_in_height_bin / num_male
p_weight_male = males_in_weight_bin / num_male
p_height_female = females_in_height_bin / num_female
p_weight_female = females_in_weight_bin / num_female

predicted_values_h = np.zeros(len(X_test))
predicted_values_w = np.zeros(len(X_test))
predicted_values_hw = np.zeros(len(X_test))

for i in range(len(X_test)):
    height_test = X_test[i, 0]
    weight_test = X_test[i, 1]

    # Find the closest bin distance for test data
    closest_height_bin = np.argmin(np.abs(height_test - height_bin_centroids))
    closest_weight_bin = np.argmin(np.abs(weight_test - weight_bin_centroids))

    # p(height | male) * p(weight | male)
    # height_hist[closest_height_bin]: number of people in that bin
    p_height = (p_height_male * prior_male) + (p_height_female * prior_female)
    p_weight = (p_weight_male * prior_male) + (p_weight_female * prior_female)

    p_male_height = (prior_male * p_height_male[closest_height_bin])/p_height[closest_height_bin]
    p_female_height = (prior_female * p_height_female[closest_height_bin])/p_height[closest_height_bin]

    p_male_weight = (prior_male * p_weight_male[closest_weight_bin])/p_weight[closest_weight_bin]
    p_female_weight = (prior_male * p_weight_female[closest_weight_bin])/p_weight[closest_weight_bin]

    posterior_male = (p_male_weight*p_male_height) / prior_male
    posterior_female = (p_female_weight*p_female_height) / prior_female

    if p_male_height > p_female_height:
        predicted_values_h[i] = 0
    else:
        predicted_values_h[i] = 1

    if p_male_weight > p_female_weight:
        predicted_values_w[i] = 0
    else:
        predicted_values_w[i] = 1

    if posterior_male > 0.5:
        predicted_values_hw[i] = 0
    else:
        predicted_values_hw[i] = 1

accuracy_h = np.mean(predicted_values_h == y_test)
accuracy_w = np.mean(predicted_values_w == y_test)
accuracy_hw = np.mean(predicted_values_hw == y_test)

print(f"Accuracy for height classifier: {accuracy_h}")
print(f"Accuracy for weight classifier: {accuracy_w}")
print(f"Accuracy for height and weight classifier: {accuracy_hw}")
