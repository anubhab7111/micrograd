# Micrograd  

Micrograd is a lightweight autograd engine for reverse-mode autodifferentiation. It uses a dynamically constructed computational graph to compute gradients for scalar operations. The project also includes a simple neural network library with a PyTorch-like API.  

Micrograd is designed to provide an understanding of how modern neural network libraries like PyTorch work under the hood. Its compact and minimalist implementation makes it an excellent starting point for learning backpropagation and autodifferentiation.

This project was inspired by Andrej Karpathy's insightful tutorial on building an autograd engine from scratch.

## Usage
```python
from micrograd.engine import Value  
from micrograd.nn import Neuron, Layer, MLP  

# Define a simple neural network  
model = MLP(3, [4, 4, 1])  # 3-inputs, two hidden layers of size 4, and 1-output  

# Example data (X: inputs, y: target)  
X = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5]]  
y = [1.0, -1.0]  

# Forward pass and loss computation  
for xi, yi in zip(X, y):  
    out = model(xi)  
    loss = (out - yi) ** 2  
    loss.backward()  

    # Gradient descent  
    for p in model.parameters():  
        p.data -= 0.01 * p.grad  
        p.grad = 0  # Reset gradients  
