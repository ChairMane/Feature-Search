import matplotlib.pyplot as plt
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

    def make_scatter(self, corr, folder, title):
        sorted_corr = sorted(corr.items(), key=lambda x: x[1])  # Something like [(frozenset({6, 1, 2}), 0.94), ....]
        featset, acc = sorted_corr[0]  # Something like featset = frozenset({6, 1, 2}), acc = 0.94
        featset = list(featset)  # Something like [6, 1, 2]



        return 1

    def make_3d_scatter(self, corr, folder, title):
        return 1