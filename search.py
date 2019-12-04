import numpy as np

class featureSearch:
    def __init__(self, data):
        self.data = data

    def leave_one_out_CV(self, X, Y, features, k, m):
        correct = 0
        current_features = list(features)

        for i in range(m):
            best = np.inf
            best_loc = -1
            rest_sum = np.zeros(m)
            distance = 0
            distance = (X[i, k] - X[:, k])**2
            for feature in current_features:
                distance += (X[i, feature] - X[:, feature])**2
            mini = np.where(distance == np.min(distance[np.nonzero(distance)]))
            if Y[mini[0][0]] == Y[i]:
                correct += 1
        return correct/m

    def backward_feature_search(self):
        Y = np.copy(self.data[:, 0])    # Classifications
        X = np.delete(self.data, 0, 1)  # Features
        #Y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
        #X = np.array([[2.7, 5.5], [8, 9.1], [0.9, 4.7], [1.1, 3.2], [5.4, 8.5], [2.9, 1.9], [6.1, 6.6], [0.5, 1], [8.3, 6.6], [8.1, 4.7]])
        current_set = set()
        (m, n) = X.shape
        accuracies = []
        correspond = {}
        default_rate = np.sum(Y == 1)
        if default_rate > len(Y) - default_rate:
            accuracies.append(default_rate/len(Y))
        else:
            accuracies.append((len(Y) - default_rate)/len(Y))
        for i in range(n):
            print('On level {} of the search tree.'.format(i+1))
            feature_to_add = -1
            best = 0
            for k in range(n):
                if k not in current_set:
                    print('--Considering adding feature {}'.format(k+1))
                    accuracy = self.leave_one_out_CV(X, Y, current_set, k, m)

                    if accuracy > best:
                        best = accuracy
                        feature_to_add = k
            accuracies.append((feature_to_add+1, best))
            current_set.add(feature_to_add)
            print('Added feature {} on level {}'.format(feature_to_add+1, i+1))
        return accuracies
    
    def forward_feature_search(self):
        Y = np.copy(self.data[:, 0])    # Classifications
        X = np.delete(self.data, 0, 1)  # Features
        #Y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
        #X = np.array([[2.7, 5.5], [8, 9.1], [0.9, 4.7], [1.1, 3.2], [5.4, 8.5], [2.9, 1.9], [6.1, 6.6], [0.5, 1], [8.3, 6.6], [8.1, 4.7]])
        current_set = set()
        (m, n) = X.shape
        accuracies = []
        correspond = {}
        default_rate = np.sum(Y == 1)
        if default_rate > len(Y) - default_rate:
            accuracies.append(default_rate/len(Y))
        else:
            accuracies.append((len(Y) - default_rate)/len(Y))
        for i in range(n):
            print('On level {} of the search tree.'.format(i+1))
            feature_to_add = -1
            best = 0
            for k in range(n):
                if k not in current_set:
                    print('--Considering adding feature {}'.format(k+1))
                    accuracy = self.leave_one_out_CV(X, Y, current_set, k, m)

                    if accuracy > best:
                        best = accuracy
                        feature_to_add = k
            accuracies.append((feature_to_add+1, best))
            current_set.add(feature_to_add)
            print('Added feature {} on level {}'.format(feature_to_add+1, i+1))
        return accuracies


if __name__ == '__main__':
    filename = input('Enter file name:')
    data = np.genfromtxt(filename)
    search = featureSearch(data)
    accuracies = search.forward_feature_search()
    print(accuracies)
