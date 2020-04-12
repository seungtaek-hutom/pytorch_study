import torch
from torch.autograd import Variable


x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]))
y_data = Variable(torch.Tensor([[0.], [0.], [0.], [1.], [1.], [1.]]))


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
    
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model()


learning_rate = 0.01
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs = 1000

for epoch in range(num_epochs):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    print(loss.data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_var = Variable(torch.Tensor([[1.0]]))
print('predict 1 hour ', 2.0, model(hour_var).data[0][0])
hour_var = Variable(torch.Tensor([[7.0]]))
print('predict 7 hour ', 7.0, model(hour_var).data[0][0])
