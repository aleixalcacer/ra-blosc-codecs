import torch


# define a base No-negative Matrix Factorization model
class NMF(torch.nn.Module):
    def __init__(self, m, n, k):
        super(NMF, self).__init__()
        self._W = torch.nn.Parameter(torch.rand(m, k), requires_grad=True)
        self._H = torch.nn.Parameter(torch.rand(k, n), requires_grad=True)

    def forward(self):
        return self.W @ self.H

    def loss(self, X):
        return torch.norm(X - self.forward()) ** 2

    def update_H(self, X):
        pass

    def fit(self, X, lr=0.01, epochs=1000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.update_H(X)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss(X)
            loss.backward()
            optimizer.step()
            self.update_H(X)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss {loss.item()}')

    def predict(self):
        return self.forward()

    @property
    def W(self):
        return self._W

    @property
    def H(self):
        return self._H


# Create a Fuzzy K-Means model

class FuzzyKMeans(NMF):

    def __init__(self, m, n, k):
        super(FuzzyKMeans, self).__init__(m, n, k)
        del self._H
        self._H = None

    def update_H(self, X):
        B = torch.pinverse(self.W.T @ self.W) @ self.W.T
        self._H = B @ X

    @property
    def W(self):
        return torch.softmax(self._W, dim=1)



# Create Archetypal Analysis model

class ArchetypalAnalysis(NMF):

    def __init__(self, m, n, k):
        super(ArchetypalAnalysis, self).__init__(m, n, k)
        del self._H
        self._B = torch.nn.Parameter(torch.rand(k, m), requires_grad=True)
        self._H = None

    def update_H(self, X):
        self._H = self.B @ X

    @property
    def W(self):
        return torch.softmax(self._W, dim=1)

    @property
    def B(self):
        return torch.softmax(self._B, dim=1)
