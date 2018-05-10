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

dataset = np.genfromtxt('/home/robert/csv_etc/mid_values.txt', delimiter =',')
datatorch = torch.from_numpy(dataset).float()
test_x = torch.cat((datatorch[142:203],datatorch[320:369]), 0)

test_y = torch.ones(61, 1)
test_y = torch.cat((test_y, torch.zeros(49, 1)), 0)
model = stair_net(376,10,1)
model.load_state_dict(torch.load('testmodel.pt'))
#model = torch.load('testmodel.pt')#['state_dict']

y_test = model.forward(test_x)

stcorr=0
nocorr=0
incorr=0
for i in range(len(test_y)):
	if y_test[i] > 0.9 and test_y[i] == 1:
		stcorr = stcorr+1
	if y_test[i] < 0.3 and test_y[i] == 0:
		nocorr = nocorr+1 
	if((y_test[i] > 0.9) and (test_y[i] == 0)) or ((y_test[i] < 0.3) and (test_y[i] == 1)):
		incorr = incorr+1
print("stairs correctly classified : %d" % (stcorr))
print("empty correctly classified : %d" % (nocorr))
print("incorrect classification : %d" % (incorr))