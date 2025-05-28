import torch
import torch.nn as nn

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets

# convert to tensors
X = torch.tensor(X.values, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)

# split into train and test, about ~5000 train and ~1500 test
X_train = X[:5000]
y_train = y[:5000]
X_test = X[5000:]
y_test = y[5000:]
  
# metadata 
# print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 

# Model ---------------------------------------------------------------------
class SimpleModel(nn.Module):
  def __init__(self):
      super(SimpleModel, self).__init__()

      # define layers
      self.layer1 = nn.Linear(11, 100, bias=False)
      self.layer2 = nn.Linear(100, 1, bias=False)
      self.activation_func = nn.ReLU()

  def forward(self, X):
      X = X.float()

      # process
      X = self.layer1(X)
      X = self.activation_func(X)
      X = self.layer2(X)

      return X

def loss_fn(preds, truth):
  func = nn.MSELoss()

  return func(preds.flatten(), truth.flatten().float())

def train_1_step(model, X, y, optim):

  model.train()
  preds = model.forward(X)
  loss = loss_fn(preds, y)

  optim.zero_grad()  # Clear old gradients
  loss.backward()
  optim.step()       # Update model parameters

  return loss.item()

# Training loop ------------------------------------------------------------
num_steps = 10000

model = SimpleModel()
optim = torch.optim.Adam(model.parameters())

for step in range(num_steps):
  loss = train_1_step(model, X_train, y_train, optim)
  print(f"Loss for step {step} is {loss}") if step % 1000 == 0 else None

# Testing -------------------------------------------------------------------
model.eval()
with torch.no_grad():
  test_preds = model(X_test)
  test_loss = loss_fn(test_preds, y_test)
  print(f"Test loss: {test_loss}")