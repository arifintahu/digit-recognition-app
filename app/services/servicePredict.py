import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.pool = nn.MaxPool2d(2, stride=2)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, 10)
		self.dropout = nn.Dropout(0.2)
		
	def forward(self, input):
		x = self.pool(F.relu(self.conv1(input)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 64 * 7 * 7)
		x = self.dropout(x)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = F.log_softmax(self.fc2(x), dim=1)
		return x

def predict(data):
	model = torch.load("./model/saved_model.pth")
	with torch.no_grad():
		logps = model(data)

	ps = torch.exp(logps)
	probab = list(ps.numpy()[0])
	return probab.index(max(probab))
