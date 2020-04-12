import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1) #one input, one output

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

#hyper-parameters
criterion = torch.nn.MSELoss(reduction='sum')
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 200


# training loop
for epoch in range(num_epochs):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# after training
hour_var = Variable(torch.Tensor([4.0]))
print("predict (after training)", 4, model.forward(hour_var).data.item())