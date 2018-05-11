import torch
import numpy as np

class stair_net(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(stair_net, self).__init__()
		self.linear1 = torch.nn.Linear(D_in, H)
		self.linear2 = torch.nn.Linear(H, D_out)

	def forward(self, x):
		sigm = torch.nn.Sigmoid()
		h = self.linear1(x)
		h_sigm = sigm(h)
		y_pred = self.linear2(h_sigm)
		return y_pred

dataset = np.genfromtxt('./processedData.txt', delimiter =',')
size = len(dataset)
split = int(0.7*size)

datatorch = torch.from_numpy(dataset).float()

test = datatorch[split:,:]
test_x, test_y = test[:, :-3], test[:, -3:]
print(test_x.shape)
print(test_y.shape)
model = stair_net(376,10,3)
model.load_state_dict(torch.load('testmodel.pt'))

y_test = model.forward(test_x)

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
print("incorrect classification : %d" % (incorr))

#print(y_test)