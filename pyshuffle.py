import numpy as np
import os




data = np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/04-05-downst1.txt', delimiter=',')
print("Non-Shuffled data:")
print(data)

data = data[~np.isinf(data).any(axis=1)]
np.random.shuffle(data)
print("Shuffled data:\n")
print(data)

#np.savetxt("testshuffle.txt", data, delimiter=',', fmt='%.3f')

np.savetxt("/home/fredde/git-repos/stairNet/testshuffle.txt", data, delimiter=',', fmt='%.3f')

#test