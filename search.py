import numpy as np
import math

def leave_one_out_CV(X, Y, features, m):
    correct = 0
    current_features = list(features)

    for i in range(m):
        best = np.inf
        best_loc = -1
        for j in range(m):
            if i != j:
                distance = 0
                # I believe the problem here is that I am iterating through
                # features at once, instead of per exemplar.
                # Might have to trace this to find out what exactly is happening.
                for k in current_features:
                    distance += (X[i, k] - X[j, k])**2
                if distance < best:
                    best = distance
                    best_loc = j
        if Y[i] == Y[best_loc]:
            correct += 1
    #print(correct)
    return correct/m

def feature_search(data):
    #Y = np.copy(data[:, 0])    # Classifications
    #X = np.delete(data, 0, 1)  # Features
    Y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    X = np.array([[2.7, 5.5], [8, 9.1], [0.9, 4.7], [1.1, 3.2], [5.4, 8.5], [2.9, 1.9], [6.1, 6.6], [0.5, 1], [8.3, 6.6], [8.1, 4.7]])
    current_set = set()
    (m, n) = X.shape
    accuracies = []
    for i in range(n):
        print('On level {} of the search tree.'.format(i+1))
        feature_to_add = -1
        best = 0
        for k in range(n):
            if k not in current_set:
                print('--Considering adding feature {}'.format(k+1))
                accuracy = leave_one_out_CV(X, Y, current_set, m)

                if accuracy > best:
                    best = accuracy
                    feature_to_add = k
        accuracies.append(best)
        current_set.add(feature_to_add)
        print('Added feature {} on level {}'.format(feature_to_add+1, i+1))
        print(accuracies)


if __name__ == '__main__':
    data = np.genfromtxt('CS170_SMALLtestdata__19.txt')
    feature_search(data)
