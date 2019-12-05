import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class featGraph:
    def __init__(self, data):
        self.data = data

    def make_bars(self, folder, title, reverse=None):
        accuracies = []
        feats = []
        for feat, acc in self.data:
            feats.append(feat)
            accuracies.append(acc)
        if title.find('LARGE') != -1:
            fig, ax = plt.subplots(figsize=(25, 15))
        else:
            fig, ax = plt.subplots(figsize=(20, 10))
        if reverse == None:
            rects = ax.bar(range(len(feats)), accuracies)
            plt.ylabel('Accuracy')
            plt.xlabel('Features')
            plt.title(title)
        else:
            rects = ax.bar(range(len(feats)), accuracies)
            plt.xticks(range(len(feats)), range(len(feats))[::-1])
            plt.ylabel('Accuracy')
            plt.xlabel('Features')
            plt.title(title)

        for rect, label in zip(rects, feats):
            height = rect.get_height()
            ax.text(rect.get_x(), height, label, ha='center', va='bottom')

        plt.savefig('charts/{}/{}.png'.format(folder, title))
        plt.close()

    def make_scatter(self, corr, folder, title, Y, X):
        sorted_corr = sorted(corr.items(), key=lambda x: x[1], reverse=True)  # Something like [(frozenset({6, 1, 2}), 0.94), ....]
        print('sorted corr is', sorted_corr)
        featset, acc = sorted_corr[0]  # Something like featset = frozenset({6, 1, 2}), acc = 0.94
        featset = list(featset)  # Something like [6, 1, 2]
        print('featset is', featset)
        x_ax = X[:, featset[2]]  # Something like feature 6 column
        y_ax = X[:, featset[1]]  # Something like feature 1 column
        colors = ['red' if i == 1 else 'blue' for i in Y]

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.scatter(x_ax, y_ax, color=colors)
        plt.ylabel('Feature {}'.format(featset[1] + 1))
        plt.xlabel('Feature {}'.format(featset[2] + 1))
        plt.title(title + ' ' + str(featset[0]+1) + ' and ' + str(featset[1]+1))

        plt.savefig('charts/{}/{}.png'.format(folder, title + ' ' + str(featset[0]+1) + ' and ' + str(featset[1]+1)))
        plt.close()

    def make_3d_scatter(self, corr, folder, title, Y, X):
        sorted_corr = sorted(corr.items(), key=lambda x: x[1], reverse=True)  # Something like [(frozenset({6, 1, 2}), 0.94), ....]
        print('sorted corr is', sorted_corr)
        featset, acc = sorted_corr[0]  # Something like featset = frozenset({6, 1, 2}), acc = 0.94
        featset = list(featset)  # Something like [6, 1, 2]
        print('featset is', featset)
        x_ax = X[:, featset[0]]  # Something like feature 6 column
        y_ax = X[:, featset[1]]  # Something like feature 1 column
        z_ax = X[:, featset[2]]
        colors = ['red' if i == 1 else 'blue' for i in Y]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.scatter(x_ax, y_ax, z_ax, color=colors, marker='o')
        ax.set_ylabel('Feature {}'.format(featset[1] + 1))
        ax.set_xlabel('Feature {}'.format(featset[0] + 1))
        ax.set_zlabel('Feature {}'.format(featset[2] + 1))
        plt.title(title + ' ' + str(featset[0]+1) + ', ' + str(featset[1]+1) + ' and ' + str(featset[2]+1))

        plt.savefig('charts/{}/{}.png'.format(folder, title + ' ' + str(featset[0]+1) + ', ' + str(featset[1]+1) + ' and ' + str(featset[2]+1)))
        plt.close()