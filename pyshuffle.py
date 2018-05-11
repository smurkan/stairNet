import numpy as np
import os




data = np.genfromtxt('./mid_values.txt', delimiter=',')
print("Non-Shuffled data:")
print(data)


np.random.shuffle(data)
print("Shuffled data:\n")
print(data)
np.savetxt("testshuffle.txt", data, delimiter=',', fmt='%.3f')
