import ssl

ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
import matplotlib.pyplot as plot_tree

fruits= np.load('fruits_300_data.npy')
print(fruits,shape)