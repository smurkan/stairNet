import numpy as np
import os


data = np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/downst1.txt', delimiter=',')
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/downst2.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/downst3.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/downst4.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/open1.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/open2.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/open3.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/open4.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/upst1.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/upst2.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/upst3.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/upst4.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/upst5.txt', delimiter=',')))
data = np.vstack((data,np.genfromtxt('/home/fredde/git-repos/stairNet/datasets/upst6.txt', delimiter=',')))
print("Non-Shuffled data:")
print(data)

data = data[~np.isinf(data).any(axis=1)]
np.random.shuffle(data)
print("Shuffled data:\n")
print(data)

#np.savetxt("testshuffle.txt", data, delimiter=',', fmt='%.3f')

np.savetxt("/home/fredde/git-repos/stairNet/processedData.txt", data, delimiter=',', fmt='%.3f')

print(np.shape(data))

#test