import numpy as np

class sampleTraining:
    def __init__(self, data, datasize, size_of_removal):
        self.data = data
        self.datasize = datasize
        self.size_of_removal = size_of_removal

    def remove_some_data(self):
        randrange = np.random.randint(self.datasize+1, size=self.size_of_removal)
        return np.delete(self.data, randrange, axis=0)