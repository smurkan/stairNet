import torch
import pandas as pd
import numpy as np

#class defining the architecture of the ANN
class stair_net(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(stair_net, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)
#feed forward part, sigmoid activation function
	def forward(self, x):
		sigm = torch.nn.Sigmoid()
		h = self.linear1(x)
		h_sigm = sigm(h)
		y_pred = self.linear2(h_sigm)
		return y_pred

#D_in is input dimension;
#H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 376, 300, 3
#read dataset as numpy array
dataset = np.genfromtxt('./processedData.txt', delimiter =',')
size = len(dataset)
#for splitting into test and training sets
split = int(0.7*size)
#convert from numpy to torch to use torch functions
datatorch = torch.from_numpy(dataset).float()
#train and test datasets
train = datatorch[:split,:]
test = datatorch[split:,:]
train_x, train_y = train[:, :-3], train[:, -3:]
test_x, test_y = test[:, :-3], test[:, -3:]
#model is our network
model = stair_net(D_in, H, D_out)
#Mean Squared Error (MSE) loss function.
loss_fn = torch.nn.MSELoss(size_average=False)
#Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(10000):
    # Forward pass: make a prediction y_pred from training data train_x
    y_pred = model(train_x)
    # Compute and print loss for every 1000th iteration
    loss = loss_fn(y_pred, train_y)
    if t%1000 == 0:
    	print(t, loss.item())

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backward pass and update weights
    loss.backward()
    optimizer.step()
#save the model to use in another script
torch.save(model.state_dict(), 'testmodel.pt')
#run test set and print results
y_test = model(test_x)
upstcorr=0
downstcorr=0
opencorr=0
incorr=0
for i in range(len(test_y)):
    if test_y[i][0] == 1 and y_test[i][0] > y_test[i][1] and y_test[i][0] > y_test[i][2] and y_test[i][0] > 0.8:  
        upstcorr = upstcorr+1
    if test_y[i][1] == 1 and y_test[i][1] > y_test[i][0] and y_test[i][1] > y_test[i][2] and y_test[i][1] > 0.8:   
        downstcorr = downstcorr+1
    if test_y[i][2] == 1 and y_test[i][2] > y_test[i][0] and y_test[i][2] > y_test[i][1] and y_test[i][2] > 0.8:    
        opencorr = opencorr+1
    if (test_y[i][0] == 1 and (y_test[i][1] > 0.8 or y_test[i][2] > 0.8)) or (test_y[i][1] == 1 and (y_test[i][0] > 0.8 or y_test[i][2] > 0.8)) or (test_y[i][2] == 1 and (y_test[i][0] > 0.8 or y_test[i][1] > 0.8)):
        incorr = incorr+1
print("upstairs correctly classified : %d" % (upstcorr))
print("downstairs correctly classified : %d" % (downstcorr))
print("open space classification : %d" % (opencorr))
print("not classified or incorrect : %d" % (2061-upstcorr-downstcorr-opencorr-incorr))

