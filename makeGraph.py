import matplotlib.pyplot as plt

class featGraph:
    def __init__(self, data):
        self.data = data

    def make_bars(self, title, reverse=None):
        accuracies = []
        feats = []
        for feat, acc in self.data:
            feats.append(feat)
            accuracies.append(acc)

        fig, ax = plt.subplots()
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

        plt.savefig('charts/{}.png'.format(title))
        plt.clf()

