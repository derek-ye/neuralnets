import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
# print(wine_quality.variables) 
# print(X.shape)

# Model ---------------------------------------------------------------------
class SimpleModel(nn.Module):
  def __init__(self):
      super(SimpleModel, self).__init__()

      # define layers
      self.layer1 = nn.Linear(11, 128)
      self.layer2 = nn.Linear(128, 64)
      self.layer3 = nn.Linear(64, 32)
      self.layer4 = nn.Linear(32, 1)
      self.activation_func = nn.ReLU()

  def forward(self, X):
      X = X.float()

      # process
      X = self.layer1(X)
      X = self.activation_func(X)
      X = self.layer2(X)
      X = self.activation_func(X)
      X = self.layer3(X)
      X = self.activation_func(X)
      X = self.layer4(X)

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
optim = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(X_train, y_train)  # Pairs up X and y
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Splits into batches of 64

for epoch in range(100):  # 100 full passes through data
    for batch_X, batch_y in dataloader:  # This loop runs ~156 times (5000/32)
        # batch_X is 32 examples, batch_y is 32 labels
        loss = train_1_step(model, batch_X, batch_y, optim)  # Process 64 examples
        # Update weights after each batch of 32

# Testing -------------------------------------------------------------------
model.eval()
with torch.no_grad():
  test_preds = model(X_test)
  test_loss = loss_fn(test_preds, y_test)
  print(f"Test loss: {test_loss}")
