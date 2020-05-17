import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 6.0, 9.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

print("predict (before training)", 4, forward(4))

for epoch in range(20):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.data[0], w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        #manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

print("predict (after training)", 4, forward(4).data[0])
