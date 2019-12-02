import numpy as np

def feature_search(data):
    Y = np.copy(data[:, 0])    # Classifications
    X = np.delete(data, 0, 1)  # Features
    # What to do after this:
    # Try understanding how you're gonna try all features.
    # So, first for loop should go through all features
    # Inner for loop should try all remaining features not in the set
    # At the end of this program, a set should have all features added.

if __name__ == '__main__':
    data = np.genfromtxt('CS170_SMALLtestdata__19.txt')
    feature_search(data)
