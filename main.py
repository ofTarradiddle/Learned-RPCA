import torch
import torch.nn as nn
import torch.optim as optim

# Define the custom RPCA loss function
class RPCALoss(nn.Module):
    def __init__(self, lambda_):
        super(RPCALoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, X, L, S):
        reconstruction_loss = torch.norm(X - (L + S), p='fro') ** 2
        sparsity_loss = self.lambda_ * torch.norm(S, p=1)
        return reconstruction_loss + sparsity_loss

# Define the RPCA model
class RPCA(nn.Module):
    def __init__(self, lambda_):
        super(RPCA, self).__init__()
        self.lambda_ = nn.Parameter(torch.tensor(lambda_))

    def forward(self, X):
        # SVD decomposition
        U, S, V = torch.svd(X)
        L = torch.mm(U, torch.diag(S))
        # Soft-thresholding to obtain sparse matrix S
        S = torch.max(S - self.lambda_, torch.zeros_like(S))
        S = torch.mm(torch.diag(S), V)
        return L, S

# Initialize the model and optimizer
lambda_ = 1
model = RPCA(lambda_)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = RPCALoss(lambda_)

# Training loop
for i in range(1000):
    L, S = model(X)
    loss = criterion(X, L, S)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
