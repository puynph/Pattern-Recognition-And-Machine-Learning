import numpy as np
import matplotlib.pyplot as plt

X_test = np.loadtxt('male_female_X_test.txt')
y_test = np.loadtxt('male_female_y_test.txt')
y_train = np.loadtxt('male_female_y_train.txt')


# random Classifier
def random_classifier(X_test_data, y_test_data):
    # generate random labels (0 or 1) for each sample
    N = len(X_test_data)
    random_labels = np.random.randint(2, size=N)
    accuracy = np.mean(random_labels == y_test_data)
    return accuracy


def most_likely_class(y_train_data, y_test_data):
    # Initialize an array to store predictions
    N_train = len(y_train_data)
    N_test = len(y_test_data)

    n_male_train = np.sum(y_train_data == 0)
    n_female_train = N_train - n_male_train

    n_male_test = np.sum(y_test_data == 0)
    n_female_test = N_test - n_male_test

    # accuracy is returned as follows since all samples are assigned to only 1 class
    if n_female_train > n_male_train:
        return n_female_test / N_test
    else:
        return n_male_test / N_test


print(f"Accuracy for random classifier: {random_classifier(X_test, y_test)}")
print(f"Accuracy for highest priori classifier: {most_likely_class(y_train, y_test)}")



