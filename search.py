import numpy as np
from makeGraph import featGraph
from training import sampleTraining

class featureSearch:
    def __init__(self, data):
        self.data = data

    def leave_one_out_CV(self, X, Y, features, k, m):
        correct = 0
        current_features = list(features)
        for i in range(m):
            distance = 0
            if k == None:
                distance = 0
            else:
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
        (m, n) = X.shape
        current_set = {s for s in range(n)}
        accuracies = []
        correspond = {}
        for i in range(n, 0, -1):
            print('On level {} of the search tree.'.format(i+1))
            feature_to_subtract = -1
            best = 0
            if i == n:
                accuracy = self.leave_one_out_CV(X, Y, current_set, None, m)
                correspond[frozenset(current_set)] = accuracy
                accuracies.append((-1, accuracy))
            else:
                for k in range(n):
                    if k in current_set:
                        current_copy = set()
                        current_copy = current_set.copy()
                        current_copy.remove(k)
                        print('--Considering removing feature {}'.format(k+1))
                        accuracy = self.leave_one_out_CV(X, Y, current_copy, None, m)

                        if accuracy > best:
                            best = accuracy
                            feature_to_subtract = k
                current_set.remove(feature_to_subtract)
                correspond[frozenset(current_set)] = best
                accuracies.append((feature_to_subtract+1, best))
                print('Removed feature {} on level {}'.format(feature_to_subtract+1, i+1))

        default_rate = np.sum(Y == 1)
        if default_rate > m - default_rate:
            correspond[frozenset()] = default_rate/m
            accuracies.append((0, default_rate / m))
        else:
            correspond[frozenset()] = (m - default_rate) / m
            accuracies.append((0, (m - default_rate) / m))
        return accuracies, correspond

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
        if default_rate > m - default_rate:
            correspond[frozenset()] = default_rate / m
            accuracies.append((0, default_rate/m))
        else:
            correspond[frozenset()] = (m - default_rate) / m
            accuracies.append((0, (m-default_rate)/m))
        print('When no elements exist in the set, the accuracy is: {:.1f}%\n\n'.format(accuracies[0][1]*100))
        print('Beginning search. . .\n')
        for i in range(n):
            feature_to_add = -1
            best = 0
            for k in range(n):
                if k not in current_set:
                    accuracy = self.leave_one_out_CV(X, Y, current_set, k, m)
                    if len(current_set) == 0:
                        print('\t\tUsing feature(s) {{{}}}: accuracy is {:.1f}%'.format(k+1, accuracy*100))
                    else:
                        print('\t\tUsing feature(s) {} and {{{}}} : accuracy is {:.1f}%'.format(current_set, k+1, accuracy*100))
                    if accuracy > best:
                        best = accuracy
                        feature_to_add = k
            current_set.add(feature_to_add)
            correspond[frozenset(current_set)] = best
            accuracies.append((feature_to_add+1, best))
            print('\nFeature(s) {} was best, with accuracy {:.1f}%\n'.format({s+1 for s in current_set}, best*100))
        sorted_corr = sorted(correspond.items(), key=lambda x: x[1], reverse=True)
        best_set, best_acc = sorted_corr[0]
        print('The best feature subset is {}, which has an accuracy of {:.1f}%'.format({s+1 for s in best_set}, best_acc*100))
        return accuracies, correspond, Y, X


if __name__ == '__main__':
    print('Welcome to Chris\' feature selection!')
    filename = input('Enter file name:')
    print('Which algorithm would you like to run? \n')
    print('1) Forward Search')
    print('2) Backward Search')
    print('3) Chris\' Special Search')
    choice = input()
    data = np.genfromtxt(filename)
    (m, n) = data.shape
    removal_size = int(m/10)
    search = featureSearch(data)
    if choice == '1':
        print('Starting forward feature search. . .')
        print('\n\n This dataset has {} features and {} examples.'.format(n, m))
        faccuracies, fcorr, Y, X = search.forward_feature_search()
        print('Forward feature search finished.')
    elif choice == '2':
        print('Starting backward feature search. . .')
        baccuracies, bcorr = search.backward_feature_search()
        print('Backward feature search finished.')
    elif choice == '3':
        i = 1
        while len(data) > 0:
            search = featureSearch(data)
            faccuracies, fcorr, Y, X = search.forward_feature_search()
            baccuracies, bcorr = search.backward_feature_search()
            new_data = sampleTraining(data, m, removal_size)
            data = new_data.remove_some_data()
            (m, n) = data.shape