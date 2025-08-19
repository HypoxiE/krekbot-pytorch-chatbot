import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
	nn.Linear(2, 100),
	nn.Tanh(),
	nn.Linear(100, 200),
	nn.Tanh(),
	nn.Linear(200, 100),
	nn.Tanh(),
	nn.Linear(100, 1),
	nn.Sigmoid()
)

model.load_state_dict(torch.load("model.pth"))

test_x = torch.tensor([0., 0.4])

pred_y = model(test_x)
print(pred_y.item())